// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

extern crate proc_macro;
use proc_macro::{TokenStream, TokenTree};

#[derive(Clone, Copy, PartialEq, Eq)]
enum CopyType {
    U8,
    U16,
    U32,
    U64,
    Usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Unroll {
    /// 1 byte
    One,
    /// `size_of::<usize>()`
    Word,
    /// 8
    WordsInBlock,
    /// WordsInBlock * Word, ie. cache line size
    Block,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Direction {
    Up,
    Down,
}

fn skip_comma(s: &mut proc_macro::token_stream::IntoIter) {
    if let Some(TokenTree::Punct(punct_item)) = s.next() {
        match punct_item.to_string().as_str() {
            "," => {}
            _ => panic!("Unexpected punctuation '{}': expected a ','.", punct_item),
        };
    } else {
        panic!("Missing comma after argument");
    };
}

fn parse_inputs(item: TokenStream) -> (CopyType, Unroll, Direction) {
    let t: CopyType;
    let unroll: Unroll;
    let direction: Direction;
    let mut item = item.into_iter();
    if let Some(TokenTree::Ident(type_item)) = item.next() {
        t = match type_item.to_string().as_str() {
            "u8" => CopyType::U8,
            "u16" => CopyType::U16,
            "u32" => CopyType::U32,
            "usize" => CopyType::Usize,
            _ => panic!(
                "Unexpected type parameter '{}': expected a 'u8', 'u16', 'u32', or 'usize'.",
                type_item
            ),
        };
    } else {
        panic!(
            "Did not find valid type parameter in the first argument position: expected a 'u8', 'u16', 'u32', or 'usize'."
        );
    };
    skip_comma(&mut item);
    let unroll_item = item.next();
    if let Some(TokenTree::Literal(unroll_item)) = unroll_item {
        unroll = match unroll_item.to_string().as_str() {
            "1" => Unroll::One,
            _ => panic!(
                "Unexpected unroll parameter '{}': expected expected a '1', 'BLOCK_SIZE', 'WORD_SIZE', or 'WORDS_IN_BLOCK'.",
                unroll_item
            ),
        };
    } else if let Some(TokenTree::Ident(unroll_item)) = unroll_item {
        unroll = match unroll_item.to_string().as_str() {
            "BLOCK_SIZE" => Unroll::Block,
            "WORD_SIZE" => Unroll::Word,
            "WORDS_IN_BLOCK" => Unroll::WordsInBlock,
            _ => panic!(
                "Unexpected unroll parameter '{}': expected expected a '1', 'BLOCK_SIZE', 'WORD_SIZE', or 'WORDS_IN_BLOCK'.",
                unroll_item
            ),
        };
    } else {
        panic!(
            "Did not find valid unroll parameter in the second argument position: expected expected a '1', 'BLOCK_SIZE', or 'WORDS_IN_BLOCK'.",
        );
    };
    skip_comma(&mut item);
    if let Some(TokenTree::Literal(direction_item)) = item.next() {
        direction = match direction_item.to_string().as_str() {
            "\"down\"" => Direction::Down,
            "\"up\"" => Direction::Up,
            _ => panic!(
                "Unexpected direction parameter '{}': expected expected 'up' or 'down'.",
                direction_item
            ),
        };
    } else {
        panic!(
            "Did not find valid direction parameter in the third argument position: expected expected 'up' or 'down'."
        );
    };
    if item.next().is_some() {
        panic!("Unexpectedly found extra parameters in gen_copy! macro");
    }
    (t, unroll, direction)
}

#[proc_macro]
pub fn gen_copy(item: TokenStream) -> TokenStream {
    let (t, unroll, direction) = parse_inputs(item);
    let mut result = String::with_capacity(10000);
    result.push_str(
        "\
// SAFETY: dst and src are NonNull<()>; they can never be null, dangling, or unaligned.
let (src, dst) = unsafe { (&mut *src.as_ptr(), &mut *dst.as_ptr()) };
",
    );
    if t == CopyType::Usize || unroll == Unroll::Word || unroll == Unroll::Block {
        gen_copy_x86(&mut result, t, unroll, direction);
        gen_copy_x86_64(&mut result, t, unroll, direction);
        gen_copy_arm(&mut result, t, unroll, direction);
        gen_copy_aarch64(&mut result, t, unroll, direction);
    } else {
        gen_copy_x86_combined(&mut result, t, unroll, direction);
        gen_copy_arm(&mut result, t, unroll, direction);
        gen_copy_aarch64(&mut result, t, unroll, direction);
    }
    result.parse().unwrap()
}

fn perform_unroll(
    result: &mut String,
    t: CopyType,
    unroll: u8,
    direction: Direction,
    f: impl Fn(&mut String, CopyType, u8),
) {
    let range = 0..unroll;
    if direction == Direction::Up {
        for offset in range.rev() {
            f(result, t, offset);
        }
    } else {
        for offset in range {
            f(result, t, offset);
        }
    }
}

fn gen_x86_copy_asm(result: &mut String, t: CopyType, offset: u8) {
    match t {
        CopyType::U8 => {
            result.push_str(&format!(
                "\
        \"mov {{scratch}}, byte ptr [{{src}} + {offset}]\",
        \"mov byte ptr [{{dst}} + {offset}], {{scratch}}\",
"
            ));
        }
        CopyType::U16 => {
            let offset = offset * 2;
            result.push_str(&format!(
                "\
        \"mov {{scratch:x}}, word ptr [{{src}} + {offset}]\",
        \"mov word ptr [{{dst}} + {offset}], {{scratch:x}}\",
"
            ));
        }
        CopyType::U32 => {
            let offset = offset * 4;
            result.push_str(&format!(
                "\
        \"mov {{scratch:e}}, dword ptr [{{src}} + {offset}]\",
        \"mov dword ptr [{{dst}} + {offset}], {{scratch:e}}\",
"
            ));
        }
        CopyType::U64 => {
            let offset = offset * 8;
            result.push_str(&format!(
                "\
        \"mov {{scratch:r}}, qword ptr [{{src}} + {offset}]\",
        \"mov qword ptr [{{dst}} + {offset}], {{scratch:r}}\",
"
            ));
        }
        CopyType::Usize => unreachable!("Resolve usize before calling gen_copy_asm"),
    }
}

fn gen_arm_copy_asm(result: &mut String, t: CopyType, offset: u8) {
    match t {
        CopyType::U8 => {
            result.push_str(&format!(
                "\
        \"ldrb {{scratch}}, [{{src}}, #{offset:#x}]\",
        \"strb {{scratch}}, [{{dst}}, #{offset:#x}]\",
"
            ));
        }
        CopyType::U16 => {
            let offset = offset * 2;
            result.push_str(&format!(
                "\
        \"ldrh {{scratch}}, [{{src}}, #{offset:#x}]\",
        \"strh {{scratch}}, [{{dst}}, #{offset:#x}]\",
"
            ));
        }
        CopyType::U32 => {
            let offset = offset * 4;
            result.push_str(&format!(
                "\
        \"ldr {{scratch}}, [{{src}}, #{offset:#x}]\",
        \"str {{scratch}}, [{{dst}}, #{offset:#x}]\",
"
            ));
        }
        CopyType::U64 => {
            unreachable!("64-bit copy not supported on ARM")
        }
        CopyType::Usize => unreachable!("Resolve usize before calling gen_copy_asm"),
    }
}

fn gen_aarch64_copy_asm(result: &mut String, t: CopyType, offset: u8) {
    match t {
        CopyType::U8 => {
            result.push_str(&format!(
                "\
            \"ldrb {{scratch:w}}, [{{src:x}}, {offset:#x}]\",
            \"strb {{scratch:w}}, [{{dst:x}}, {offset:#x}]\",
"
            ));
        }
        CopyType::U16 => {
            let offset = offset * 2;
            result.push_str(&format!(
                "\
            \"ldrh {{scratch:w}}, [{{src:x}}, {offset:#x}]\",
            \"strh {{scratch:w}}, [{{dst:x}}, {offset:#x}]\",
"
            ));
        }
        CopyType::U32 => {
            let offset = offset * 4;
            result.push_str(&format!(
                "\
            \"ldr {{scratch:w}}, [{{src:x}}, {offset:#x}]\",
            \"str {{scratch:w}}, [{{dst:x}}, {offset:#x}]\",
"
            ));
        }
        CopyType::U64 => {
            let offset = offset * 8;
            result.push_str(&format!(
                "\
            \"ldr {{scratch:x}}, [{{src:x}}, {offset:#x}]\",
            \"str {{scratch:x}}, [{{dst:x}}, {offset:#x}]\",
"
            ));
        }
        CopyType::Usize => unreachable!("Resolve usize before calling gen_copy_asm"),
    }
}

fn gen_copy_x86(result: &mut String, t: CopyType, unroll: Unroll, direction: Direction) {
    let t = match t {
        CopyType::U8 => t,
        CopyType::U16 => t,
        CopyType::U32 => t,
        CopyType::U64 => t,
        CopyType::Usize => CopyType::U32,
    };
    let unroll = match unroll {
        Unroll::One => 1,
        Unroll::Word => 4,
        Unroll::WordsInBlock => 8,
        Unroll::Block => 32,
    };
    result.push_str("#[cfg(target_arch = \"x86\")]\n");
    start_asm(result);
    perform_unroll(result, t, unroll, direction, gen_x86_copy_asm);
    end_asm(result, t == CopyType::U8);
}

fn gen_copy_x86_64(result: &mut String, t: CopyType, unroll: Unroll, direction: Direction) {
    let t = match t {
        CopyType::U8 => t,
        CopyType::U16 => t,
        CopyType::U32 => t,
        CopyType::U64 => t,
        CopyType::Usize => CopyType::U64,
    };
    let unroll = match unroll {
        Unroll::One => 1,
        Unroll::Word => 8,
        Unroll::WordsInBlock => 8,
        Unroll::Block => 64,
    };
    result.push_str("#[cfg(target_arch = \"x86_64\")]\n");
    start_asm(result);
    perform_unroll(result, t, unroll, direction, gen_x86_copy_asm);
    end_asm(result, t == CopyType::U8);
}

fn gen_copy_x86_combined(result: &mut String, t: CopyType, unroll: Unroll, direction: Direction) {
    let t = match t {
        CopyType::U8 => t,
        CopyType::U16 => t,
        CopyType::U32 => t,
        _ => unreachable!(
            "Cannot generate combined assembly when size is not static and 32-bits or below"
        ),
    };
    let unroll = match unroll {
        Unroll::One => 1,
        Unroll::WordsInBlock => 8,
        _ => unreachable!("Cannot generate combined assembly when unroll is word-dependent"),
    };
    result.push_str("#[cfg(any(target_arch = \"x86\", target_arch = \"x86_64\"))]\n");
    start_asm(result);
    perform_unroll(result, t, unroll, direction, gen_x86_copy_asm);
    end_asm(result, t == CopyType::U8);
}

fn gen_copy_arm(result: &mut String, t: CopyType, unroll: Unroll, direction: Direction) {
    let t = match t {
        CopyType::U8 => t,
        CopyType::U16 => t,
        CopyType::U32 => t,
        CopyType::U64 => t,
        CopyType::Usize => CopyType::U32,
    };
    let unroll = match unroll {
        Unroll::One => 1,
        Unroll::Word => 4,
        Unroll::WordsInBlock => 8,
        Unroll::Block => 32,
    };
    result.push_str("#[cfg(target_arch = \"arm\")]\n");
    start_asm(result);
    perform_unroll(result, t, unroll, direction, gen_arm_copy_asm);
    end_asm(result, false);
}

fn gen_copy_aarch64(result: &mut String, t: CopyType, unroll: Unroll, direction: Direction) {
    let t = match t {
        CopyType::U8 => t,
        CopyType::U16 => t,
        CopyType::U32 => t,
        CopyType::U64 => t,
        CopyType::Usize => CopyType::U64,
    };
    let unroll = match unroll {
        Unroll::One => 1,
        Unroll::Word => 8,
        Unroll::WordsInBlock => 8,
        Unroll::Block => 64,
    };
    result.push_str("#[cfg(target_arch = \"aarch64\")]\n");
    start_asm(result);
    perform_unroll(result, t, unroll, direction, gen_aarch64_copy_asm);
    end_asm(result, false);
}

fn start_asm(result: &mut String) {
    result.push_str(
        "\
unsafe {
        core::arch::asm!(
",
    );
}

fn end_asm(result: &mut String, scratch_is_reg_byte: bool) {
    if scratch_is_reg_byte {
        result.push_str(
            "\
                scratch = out(reg_byte) _,
",
        );
    } else {
        result.push_str(
            "\
                scratch = out(reg) _,
",
        );
    }
    result.push_str(
        "\
            dst = in(reg) dst,
            src = in(reg) src,
            options(preserves_flags, nostack)
    );
}
",
    );
}
