#![no_std]

use core::{ptr::NonNull, sync::atomic::AtomicU16};

use ecmascript_atomics::atomic_exchange_16_seq_cst;

static mut MEM: u64 = 0;

fn main() {
    let mem = unsafe { NonNull::new(&raw mut MEM).unwrap_unchecked() }.cast::<()>();

    let mem_start = unsafe { MEM };
    let proper_a = AtomicU16::swap(
        unsafe { mem.cast::<AtomicU16>().as_ref() },
        0xFFFF,
        core::sync::atomic::Ordering::SeqCst,
    );
    let proper_mem_a = unsafe { MEM };

    unsafe { mem.cast::<u64>().write(0x0) };
    let a = atomic_exchange_16_seq_cst(mem, 0xFFFF);
    let mem_a = unsafe { MEM };
    let b = atomic_exchange_16_seq_cst(mem, 0);
    let mem_b = unsafe { MEM };
    if mem_start != 0x0 {
        panic!("MEM didn't start with 0");
    } else if proper_a != 0x0 || proper_mem_a != 0xFFFF {
        panic!("Wrong {proper_a} 0x{proper_mem_a:x?}");
    } else if a != 0x0 || mem_a != 0xFFFF || b != 0xFFFF || mem_b != 0x0 {
        panic!("Wrong {a} 0x{mem_a:x?} {b} 0x{mem_b:x?}");
    }
}
