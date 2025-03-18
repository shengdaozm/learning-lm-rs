mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;

use std::collections::HashMap;
// use std::collections::hash_map::Entry;
use std::io::{self, Write};
use std::path::PathBuf;
use tokenizers::Tokenizer;

struct ConversationsCache {
    messages: Vec<String>,
    cache: kvcache::KVCache<f32>,
}

impl ConversationsCache {
    fn new(message_: Vec<String>, cache_: kvcache::KVCache<f32>) -> Self {
        ConversationsCache {
            messages: message_,
            cache: cache_,
        }
    }
}

fn main() {
    story();
    // chat();
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
    let output_ids = llama.generate(input_ids, 500, 0.8, 30, 1., &mut llama.new_cache());
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

// AI对话
fn chat() {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<f32>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    
    let mut conversations = HashMap::new();
    let mut current_id = 0;
    let mut next_id: i32 = 1;  // 用于生成新ID
    
    // 初始化默认对话
    conversations.insert(current_id, ConversationsCache::new(Vec::new(), llama.new_cache()));

    loop {
        // 获取当前对话的可变引用
        let conv = conversations.get_mut(&current_id).expect("对话不存在");
        let messages = &mut conv.messages;
        let cache = &mut conv.cache;

        println!("==== 当前对话ID: {} ====", current_id);
        
        loop {
            print!("用户: ");
            io::stdout().flush().unwrap();
            let mut input= String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            match input.to_lowercase().as_str() {
                "quit" => return,
                "change" => {
                    println!("== 切换对话 ==");
                    println!("当前ID: {}，输入新ID（新建输入-1）:", current_id);
                    
                    let mut new_id_input = String::new();
                    io::stdin().read_line(&mut new_id_input).unwrap();
                    let new_id = new_id_input.trim().parse::<i32>().unwrap();

                    match new_id {
                        -1 => {
                            current_id = next_id;
                            next_id = next_id.wrapping_add(1);
                            conversations.entry(current_id)
                                .or_insert_with(|| ConversationsCache::new(Vec::new(), llama.new_cache()));
                            println!("== 新建对话 {} ==", current_id);
                        },
                        _ => {
                            if conversations.contains_key(&new_id) {
                                current_id = new_id;
                                println!("== 切换到对话 {} ==", current_id);
                            } else {
                                println!("!! 对话 {} 不存在 !!", new_id);
                                return ;
                            }
                        }
                    }
                    break; // 退出内层循环，重新获取对话引用
                },
                _ => {
                    // 处理正常对话
                    messages.push(format!("<|im_start|>user\n{}<|im_end|>", input));
                    let prompt = messages.join("\n") + "\n<|im_start|>assistant\n";
                    
                    let binding = tokenizer.encode(prompt, true).unwrap();
                    let input_ids = binding.get_ids();
                    
                    let output_ids = llama.generate(input_ids, 200, 0.8, 30, 1., cache);
                    let response = tokenizer.decode(&output_ids, true).unwrap();
                    
                    messages.push(format!("<|im_start|>assistant\n{}<|im_end|>", response));
                    println!("助手: {}", response);
                }
            }
        }
    }
}
