mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::path::PathBuf;
use tokenizers::Tokenizer;
use std::io::{self, Write};

fn main() {
    //story();
    chat();
}

fn story() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);
    let output_ids = llama.generate(input_ids, 500, 0.8, 30, 1.,&mut llama.new_cache());
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

// AI对话
fn chat() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");

    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();

    let mut messages = Vec::new(); // 存储对话历史
    let mut cache = llama.new_cache(); // 初始化 KVCache

    loop {
        print!("User: ");
        io::stdout().flush().unwrap(); // 确保提示符立即显示
        let mut user_input = String::new();
        io::stdin().read_line(&mut user_input).unwrap();
        let user_input = user_input.trim();

        messages.push(format!("<|im_start|>user\n{}<|im_end|>", user_input));

        let prompt = messages.join("\n") + "\n<|im_start|>assistant\n";

        let binding = tokenizer.encode(prompt, true).unwrap();
        let input_ids = binding.get_ids();
        let output_ids = llama.generate(input_ids, 500, 0.8, 30, 1., &mut cache);
        let assistant_response = tokenizer.decode(&output_ids, true).unwrap();

        messages.push(format!("<|im_start|>assistant\n{}<|im_end|>", assistant_response));

        println!("Assistant: {}", assistant_response);
    }
}
