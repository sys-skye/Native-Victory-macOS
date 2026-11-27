//! Translates a dumped graphics piece pair as the guessed path does:
//! `VICTORY_CODE=<code> VICTORY_META=<metadata> VICTORY_STAGE=vs|ps cargo run --example guess`.

fn main() {
    let read = |k: &str| std::fs::read(std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set"))).expect("readable piece");
    let stage = match std::env::var("VICTORY_STAGE").as_deref() {
        Ok("ps") => victory_shader::HwStage::Ps,
        _ => victory_shader::HwStage::Vs,
    };
    let c = if std::env::var_os("VICTORY_CONTAINER").is_some() {
        read("VICTORY_CONTAINER")
    } else {
        victory_shader::emit::graphics::container(stage, &read("VICTORY_CODE"), &read("VICTORY_META")).expect("container")
    };
    let options = victory_shader::emit::Options { wave32_host: true, zero_copy_memory: true, rect_list: std::env::var_os("VICTORY_RECT").is_some(), ..Default::default() };
    match victory_shader::emit::translate_with(&c, options) {
        Ok(h) => {
            println!("{:?} dxil {} bytes", h.stage, h.dxil.len());
            println!("user_data {:?}", h.user_data);
            println!("table_entries {:?}", h.table_entries);
            println!("cbvs {:?} srvs {:?} uavs {:?} memory {} heap {} view_addresses {}", h.root_cbvs, h.root_srvs, h.root_uavs, h.memory, h.heap, h.view_addresses);
            if let Ok(p) = std::env::var("VICTORY_DXIL") {
                std::fs::write(p, &h.dxil).expect("writable");
            }
        }
        Err(e) => println!("translation failed: {e}"),
    }
}
