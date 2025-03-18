use std::fs::File;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators as OP;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::path::Path;

pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    eos_token_id: u32,      // end token id
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        // Embedding lookup
        OP::gather(&mut residual, input, &self.params.embedding_table);

        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            OP::matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            OP::matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
                                                       //multi-self-attention
            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );
            // down project
            OP::matmul_transb(
                &mut residual,
                1.0,
                &hidden_states,
                &self.params.wo[layer],
                1.0,
            );
            //mlp
            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        // No matter what seq_len, the output is always a 1D vector of length vocab,
        // which contains the probabilities for the next token.
        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        OP::matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: f32,
        top_k: u32,
        temperature: f32,
        cache: &mut KVCache<f32>,
    ) -> Vec<u32> {
        let mut result = Vec::<u32>::new();

        // 将输入的 token_ids 转换为 Tensor
        let mut input_tensor = Tensor::<u32>::new(token_ids.to_vec(), &vec![token_ids.len()]);

        // 按照最大长度生成结果
        for _ in 0..max_len {
            // 前向传播，获取 logits
            let logits = self.forward(&input_tensor, cache);

            // 从 logits 中采样下一个 token
            let next_token = OP::random_sample(&logits, top_p, top_k, temperature);

            if next_token == self.eos_token_id {
                break;
            }
            result.push(next_token);

            // 更新输入，用于下一次生成
            input_tensor = Tensor::<u32>::new(vec![next_token], &vec![1]);
        }

        result
    }
}

pub fn self_attention(
    hidden_states: &mut crate::tensor::Tensor<f32>, //隐藏层，[seq_len,total_head_num*d_head]
    att_scores: &mut crate::tensor::Tensor<f32>,    //注意力分数
    q: &crate::tensor::Tensor<f32>,                 //[seq_len,total_head_num*d_head]
    k: &crate::tensor::Tensor<f32>,                 //[total_seq,num_key_value_heads*d_head]
    v: &crate::tensor::Tensor<f32>,                 //[total_seq,num_key_value_heads*d_head]
    n_kv_h: usize,                                  //kv头数,减少计算量
    n_groups: usize,                                //每个 Key 组对应的 Query 头数,减少计算量
    seq_len: usize,                                 // 输入序列长度
    total_seq_len: usize,                           // 总序列长度
    dqkv: usize,                                    // head_dim
) {
    {
        let att_data = unsafe { att_scores.data_mut() };
        att_data.fill(0.0);
    }

    let q_data = q.data();
    let k_data = k.data();
    let inv_scale = 1.0 / (dqkv as f32).sqrt();

    let total_head_num = n_kv_h * n_groups; //总头数
    let total_d_q = total_head_num * dqkv; //头的总长度
    let total_d_kv = n_kv_h * dqkv; //每个kv的长度

    let total_d_atts_3 = n_groups * seq_len * total_seq_len;
    let total_d_atts_2 = seq_len * total_seq_len;
    
    {
        // 使用分块并行写入解决借用冲突
        //let total_att_elements = n_kv_h * n_groups * seq_len * total_seq_len;
        let att_data = unsafe { att_scores.data_mut() };
        
        // 将注意力矩阵按KV头分块并行处理
        att_data.par_chunks_mut(total_d_atts_3)
            .enumerate()
            .for_each(|(index_kv_h, att_block)| {
                let offset_k = index_kv_h * dqkv;
                
                // 每个KV头内部按组并行
                att_block.par_chunks_mut(total_d_atts_2)
                    .enumerate()
                    .for_each(|(curr_q_in_group, att_group_block)| {
                        let curr_att_head = index_kv_h * n_groups + curr_q_in_group;
                        let offset_q = curr_att_head * dqkv;
    
                        // 按序列位置并行
                        att_group_block.par_chunks_mut(total_seq_len)
                            .enumerate()
                            .for_each(|(i_seq, att_seq_block)| {
                                let begin_vec_q = i_seq * total_d_q + offset_q;
    
                                // 并行计算每个位置的注意力分数
                                att_seq_block.par_iter_mut().enumerate().for_each(|(i_tseq, att_val)| {
                                    let begin_vec_k = i_tseq * total_d_kv + offset_k;
                                    
                                    // 向量化点积计算（4路展开）
                                    let mut dot = 0.0;
                                    for k in (0..dqkv).step_by(4) {
                                        let end = (k + 4).min(dqkv);
                                        dot += q_data[begin_vec_q + k..begin_vec_q + end]
                                            .iter()
                                            .zip(&k_data[begin_vec_k + k..begin_vec_k + end])
                                            .map(|(&q, &k)| q * k)
                                            .sum::<f32>();
                                    }
                                    
                                    *att_val = dot * inv_scale;
                                });
                            });
                    });
            });
    }

    OP::masked_softmax(att_scores);

    let att_data = att_scores.data();
    let v_data = v.data();
    {
        let hs_data = unsafe { hidden_states.data_mut() };
        hs_data.fill(0.0);

        for index_kv_h in 0..n_kv_h {
            let offset_matrix_v_g = index_kv_h * dqkv;
            for curr_q_in_group in 0..n_groups {
                let offset_matrix_a_h =
                    curr_q_in_group * total_d_atts_2 + index_kv_h * total_d_atts_3;
                for curr_idx_seq in 0..seq_len {
                    let begin_vec_a = offset_matrix_a_h + curr_idx_seq * total_seq_len;
                    for curr_idx_dhead in 0..dqkv {
                        let begin_vec_v = curr_idx_dhead + offset_matrix_v_g;
                        let mut sum_ = 0.0;
                        for curr_idx_tseq in 0..total_seq_len {
                            let idx_a = begin_vec_a + curr_idx_tseq;
                            let idx_v = begin_vec_v + curr_idx_tseq * total_d_kv;
                            sum_ = sum_ + att_data[idx_a] * v_data[idx_v];
                        }

                        let curr_att_head =
                            index_kv_h * n_groups + curr_q_in_group;
                        let hs_offset = curr_idx_seq * (total_head_num * dqkv)
                            + curr_att_head * dqkv
                            + curr_idx_dhead;
                        hs_data[hs_offset] = sum_;
                    }
                }
            }
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,      //输入残差
    hidden_states: &mut Tensor<f32>, //隐藏层，自注意输出
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    /*
    act = gate * sigmoid(gate) * up ## SwiGLU
    output = act @ down_weight.T
    residual = output + residual
    */
    // Step 1: RMS normalization
    OP::rms_norm(hidden_states, &residual, &rms_w, eps); //hidden = rms_norm(residual)

    // Step 2: Compute gate and up branches
    OP::matmul_transb(gate, 0.0, &hidden_states, &w_gate, 1.0); // gate = hidden_states @ w_gate.T
    OP::matmul_transb(up, 0.0, &hidden_states, &w_up, 1.0); // up = hidden_states @ w_up.T

    // Step 3: SwiGLU activation
    OP::swiglu(up, &gate); // gate = gate * sigmoid(gate) * up

    // Step 4: Compute output
    OP::matmul_transb(residual, 1.0, &up, &w_down, 1.0);
}

#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use crate::tensor::float_eq;
    use std::path::PathBuf;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(
        &model.params.embedding_table.data()[50],
        &0.14453125,
        1e-6
    ));
    assert_eq!(
        model.params.lm_head.data()[10],
        model.params.embedding_table.data()[10]
    );
    assert!(float_eq(
        &model.params.rms_att_w[0].data()[10],
        &0.18652344,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_ffn_w[1].data()[10],
        &0.32421875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.rms_out_w.data()[100],
        &0.73046875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.w_down[0].data()[100],
        &-0.0625,
        1e-6
    ));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(
        &model.params.w_gate[1].data()[100],
        &0.296875,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wq[1].data()[100],
        &0.032226563,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wk[1].data()[100],
        &-0.21386719,
        1e-6
    ));
    assert!(float_eq(
        &model.params.wv[0].data()[100],
        &0.041015625,
        1e-6
    ));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));
}
