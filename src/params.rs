use std::convert;

use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    // Helper function to convert u8 slice to f32 vector
    fn u8_to_f32_vec(data: &[u8]) -> Vec<f32> {
        // 检查数据长度是否是4的倍数
        assert!(data.len() % 4 == 0, "Input &[u8] length must be a multiple of 4");
    
        // 每4字节解析为一个f32
        data.chunks(4)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().expect("Chunk size should be 4");
                f32::from_le_bytes(bytes) // 假设数据是小端序（LE）
            })
            .collect()
    }
    

    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // Helper closure to load a tensor by name
        let get_tensor = |name: &str| {
            let tensor_data = safetensor.tensor(name)
                .expect(&format!("Tensor {name} not found in safetensors file"));
            let shape = tensor_data.shape().iter().map(|&x| x as usize).collect::<Vec<_>>();            
            let float_data  = Self::u8_to_f32_vec(tensor_data.data());
            println!("get the tensor {} successfully", name);
            Tensor::new(float_data, &shape)
        };

        // Helper function to load a vector of tensors for each layer
        let load_layers = |prefix: &str, layers: usize| -> Vec<Tensor<f32>> {
            (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{}.{}", i, prefix)))
                .collect()
        };
        // test safetensor here
        for name in safetensor.names() {
            println!("Found tensor name: {}", name);
        }
        // Construct LLamaParams using get_tensor and load_layers
        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"), // 不是很理解
            rms_att_w: load_layers("input_layernorm.weight", config.num_hidden_layers),
            wq: load_layers("self_attn.q_proj.weight", config.num_hidden_layers),
            wk: load_layers("self_attn.k_proj.weight", config.num_hidden_layers),
            wv: load_layers("self_attn.v_proj.weight", config.num_hidden_layers),
            wo: load_layers("self_attn.o_proj.weight", config.num_hidden_layers),
            rms_ffn_w: load_layers("post_attention_layernorm.weight", config.num_hidden_layers),
            w_up: load_layers("mlp.up_proj.weight", config.num_hidden_layers),
            w_gate: load_layers("mlp.gate_proj.weight", config.num_hidden_layers),
            w_down: load_layers("mlp.down_proj.weight", config.num_hidden_layers),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }

    }
}
