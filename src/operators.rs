use core::panic;

use rand::distributions::Slice;

use crate::tensor::Tensor;
// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let shape_y = y.shape();
    let shape_x = x.shape();

    assert!(shape_x == shape_y);
    match shape_x.last() {
        Some(n) => {
            assert!(*n == w.size());
        }
        None => {
            panic!("shape_x must have at least one dimension");
        }
    }

    let y = unsafe { y.data_mut() };
    let len = x.size();
    let x = x.data();
    let n = *shape_x.last().unwrap();
    let mut start_index: usize = 0;
    loop {
        // option for every n elements
        if start_index >= len {
            break;
        }
        let rms: f32 = x.iter().skip(start_index).take(n).map(|&x| x * x).sum();
        let mu = (rms / (n as f32) + epsilon).sqrt() as f32;
        for i in start_index..start_index + n {
            y[i] = x[i] * w.data()[i % n] / mu;
        }
        start_index += n;
    }
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size(); //size is the tensor's length
    assert!(len == x.size());

    let y = unsafe { y.data_mut() };
    let x = x.data();

    for i in 0..len {
        y[i] = y[i] * x[i] * (1.0 / ((-x[i]).exp() + 1.0));
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    // 如下代码仅仅针对2维矩阵，高维的tensor相乘中，转置和广播都需要实现
    let shape_c = c.shape();
    let shape_a = a.shape();
    let shape_b = b.shape();
    assert!(shape_b.len() == 2 && shape_a.len() == 2 && shape_c.len() == 2);

    let c = unsafe { c.data_mut() };
    let len = c.len();
    let beta_c: Vec<f32> = c.iter().map(|&val| val * beta).collect();
    let alpha_ab=caculate2mat(a, b, alpha);
    // a-> m*k , b->n*k ,c ->m*n
    for i in 0..len {
        c[i]=beta_c[i] + alpha_ab[i] ;
    }
}

// NOTE: 用于计算2维矩阵a*b^T，无法兼容高纬的tensor
fn caculate2mat(a: &Tensor<f32>, b: &Tensor<f32>,alpha:f32) -> Vec<f32> {
    let shape_a = a.shape();
    let shape_b = b.shape();
    assert!(shape_a.len() == 2 && shape_b.len() == 2);

    let a = a.data();
    let b = b.data();
    let mut mata: Vec<Vec<f32>> = vec![vec![0.0; shape_a[1]]; shape_a[0]];
    let mut matbb: Vec<Vec<f32>> = vec![vec![0.0; shape_b[1]]; shape_b[0]];
    let mut matb: Vec<Vec<f32>> = vec![vec![0.0; shape_b[0]]; shape_b[1]];
    let m=shape_a[0];
    let n=shape_b[0];
    assert!(shape_a[1]== shape_b[1]);
    
    //生成a
    let mut index=0;
    for i in 0..shape_a[0] {
        for j in 0..shape_a[1] {
            mata[i][j]=a[index];
            index += 1;
        }
    }
    //生成b
    index = 0;
    for i in 0..shape_b[0] {
        for j in 0..shape_b[1] {
            matbb[i][j]=b[index];
            index += 1;
        }
    }
    //b 转置
    for i in 0..shape_b[1] {
        for j in 0..shape_b[0] {
            matb[i][j]=matbb[j][i];
        }
    }
    // println!("mata:{:?}",mata);
    // println!("matb:{:?}",matb);
    // print!("m:{:?} n:{:?}",m,n);
    let mut tmp_ans: Vec<Vec<f32>> = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            for k in 0..shape_a[1] {
                // meta m*k ,metab k*n
                tmp_ans[i][j] += mata[i][k] * matb[k][j]*alpha;
            }
        }
    }
    tmp_ans.into_iter().flatten().collect()
}

// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
