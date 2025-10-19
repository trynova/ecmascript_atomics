// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::ptr::NonNull;

use ecmascript_atomics::*;

#[test]
fn test_load() {
    let dst = Box::new(u64::MAX);
    let dst = unsafe { RacyMemory::enter_ptr(NonNull::from(Box::leak(dst))) };
    let dst_slice = dst.as_slice();
    let dst_u8 = dst_slice.as_aligned::<u8>().unwrap();
    let dst_u16 = dst_slice.as_aligned::<u16>().unwrap();
    let dst_u32 = dst_slice.as_aligned::<u32>().unwrap();
    let dst_u64 = dst_slice.get(0).unwrap();

    assert_eq!(dst_u8.load(Ordering::Unordered), u8::MAX);

    assert_eq!(dst_u16.load(Ordering::Unordered), u16::MAX);

    assert_eq!(dst_u32.load(Ordering::Unordered), u32::MAX);

    assert_eq!(dst_u64.load(Ordering::Unordered), u64::MAX);

    assert_eq!(dst_u8.load(Ordering::SeqCst), u8::MAX);

    assert_eq!(dst_u16.load(Ordering::SeqCst), u16::MAX);

    assert_eq!(dst_u32.load(Ordering::SeqCst), u32::MAX);

    assert_eq!(dst_u64.load(Ordering::SeqCst), u64::MAX);

    // SAFETY: Correct type.
    let _ = unsafe { Box::from_raw(dst.exit().0.as_ptr()) };
}

#[test]
fn test_store() {
    let dst = Box::new(0u64);
    let dst = unsafe { RacyMemory::enter_ptr(NonNull::from(Box::leak(dst))) };
    let dst_slice = dst.as_slice();
    let dst_u8 = dst_slice.as_aligned::<u8>().unwrap();
    let dst_u16 = dst_slice.as_aligned::<u16>().unwrap();
    let dst_u32 = dst_slice.as_aligned::<u32>().unwrap();
    let dst_u64 = dst_slice.as_aligned::<u64>().unwrap();

    dst_u8.store(u8::MAX, Ordering::Unordered);
    assert_eq!(dst_u8.load(Ordering::Unordered), u8::MAX);

    dst_u16.store(u16::MAX, Ordering::Unordered);
    assert_eq!(dst_u16.load(Ordering::Unordered), u16::MAX);

    dst_u32.store(u32::MAX, Ordering::Unordered);
    assert_eq!(dst_u32.load(Ordering::Unordered), u32::MAX);

    dst_u64.store(u64::MAX, Ordering::Unordered);
    assert_eq!(dst_u64.load(Ordering::Unordered), u64::MAX);

    dst_u8.store(0, Ordering::SeqCst);
    assert_eq!(dst_u8.load(Ordering::SeqCst), 0);

    dst_u16.store(0, Ordering::SeqCst);
    assert_eq!(dst_u16.load(Ordering::SeqCst), 0);

    dst_u32.store(0, Ordering::SeqCst);
    assert_eq!(dst_u32.load(Ordering::SeqCst), 0);

    dst_u64.store(0, Ordering::SeqCst);
    assert_eq!(dst_u64.load(Ordering::SeqCst), 0);

    // SAFETY: Correct type.
    let _ = unsafe { Box::from_raw(dst.exit().0.as_ptr()) };
}

#[test]
fn test_exchange() {
    let dst = Box::new(0u64);
    let dst = unsafe { RacyMemory::enter_ptr(NonNull::from(Box::leak(dst))) };
    let dst_slice = dst.as_slice();
    let dst_u8 = dst_slice.as_aligned::<u8>().unwrap();
    let dst_u16 = dst_slice.as_aligned::<u16>().unwrap();
    let dst_u32 = dst_slice.as_aligned::<u32>().unwrap();
    let dst_u64 = dst_slice.as_aligned::<u64>().unwrap();

    assert_eq!(dst_u8.swap(u8::MAX), 0);
    assert_eq!(dst_u8.swap(0), u8::MAX);

    assert_eq!(dst_u16.swap(u16::MAX), 0);
    assert_eq!(dst_u16.swap(0), u16::MAX);

    assert_eq!(dst_u32.swap(u32::MAX), 0);
    assert_eq!(dst_u32.swap(0), u32::MAX);

    assert_eq!(dst_u64.swap(u64::MAX), 0);
    assert_eq!(dst_u64.swap(0), u64::MAX);

    // SAFETY: Correct type.
    let _ = unsafe { Box::from_raw(dst.exit().0.as_ptr()) };
}

#[test]
fn test_compare_exchange() {
    let dst = Box::new(0u64);
    let dst = unsafe { RacyMemory::enter_ptr(NonNull::from(Box::leak(dst))) };
    let dst_slice = dst.as_slice();
    let dst_u8 = dst_slice.as_aligned::<u8>().unwrap();
    let dst_u16 = dst_slice.as_aligned::<u16>().unwrap();
    let dst_u32 = dst_slice.as_aligned::<u32>().unwrap();
    let dst_u64 = dst_slice.as_aligned::<u64>().unwrap();

    assert_eq!(dst_u8.compare_exchange(u8::MAX, u8::MAX), Err(0));
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);
    assert_eq!(dst_u8.compare_exchange(0, u8::MAX), Ok(0));
    assert_eq!(dst_u64.load(Ordering::Unordered), u8::MAX as u64);
    assert_eq!(dst_u8.compare_exchange(0, 0), Err(u8::MAX));
    assert_eq!(dst_u64.load(Ordering::Unordered), u8::MAX as u64);
    assert_eq!(dst_u8.compare_exchange(u8::MAX, 0), Ok(u8::MAX));
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);

    assert_eq!(dst_u16.compare_exchange(u16::MAX, u16::MAX), Err(0));
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);
    assert_eq!(dst_u16.compare_exchange(0, u16::MAX), Ok(0));
    assert_eq!(dst_u64.load(Ordering::Unordered), u16::MAX as u64);
    assert_eq!(dst_u16.compare_exchange(0, 0), Err(u16::MAX));
    assert_eq!(dst_u64.load(Ordering::Unordered), u16::MAX as u64);
    assert_eq!(dst_u16.compare_exchange(u16::MAX, 0), Ok(u16::MAX));
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);

    assert_eq!(dst_u32.compare_exchange(u32::MAX, u32::MAX), Err(0));
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);
    assert_eq!(dst_u32.compare_exchange(0, u32::MAX), Ok(0));
    assert_eq!(dst_u64.load(Ordering::Unordered), u32::MAX as u64);
    assert_eq!(dst_u32.compare_exchange(0, 0), Err(u32::MAX));
    assert_eq!(dst_u64.load(Ordering::Unordered), u32::MAX as u64);
    assert_eq!(dst_u32.compare_exchange(u32::MAX, 0), Ok(u32::MAX));
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);

    assert_eq!(dst_u64.compare_exchange(u64::MAX, u64::MAX), Err(0));
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);
    assert_eq!(dst_u64.compare_exchange(0, u64::MAX), Ok(0));
    assert_eq!(dst_u64.load(Ordering::Unordered), u64::MAX);
    assert_eq!(dst_u64.compare_exchange(0, 0), Err(u64::MAX));
    assert_eq!(dst_u64.load(Ordering::Unordered), u64::MAX);
    assert_eq!(dst_u64.compare_exchange(u64::MAX, 0), Ok(u64::MAX));
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);

    // SAFETY: Correct type.
    let _ = unsafe { Box::from_raw(dst.exit().0.as_ptr()) };
}

#[test]
fn test_add() {
    let dst = Box::new(0u64);
    let dst = unsafe { RacyMemory::enter_ptr(NonNull::from(Box::leak(dst))) };
    let dst_slice = dst.as_slice();
    let dst_u8 = dst_slice.as_aligned::<u8>().unwrap();
    let dst_u16 = dst_slice.as_aligned::<u16>().unwrap();
    let dst_u32 = dst_slice.as_aligned::<u32>().unwrap();
    let dst_u64 = dst_slice.as_aligned::<u64>().unwrap();

    assert_eq!(dst_u8.fetch_add(u8::MAX), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), u8::MAX as u64);
    assert_eq!(dst_u8.fetch_add(1), u8::MAX);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);

    assert_eq!(dst_u16.fetch_add(u16::MAX), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), u16::MAX as u64);
    assert_eq!(dst_u16.fetch_add(1), u16::MAX);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);

    assert_eq!(dst_u32.fetch_add(u32::MAX), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), u32::MAX as u64);
    assert_eq!(dst_u32.fetch_add(1), u32::MAX);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);

    assert_eq!(dst_u64.fetch_add(u64::MAX), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), u64::MAX);
    assert_eq!(dst_u64.fetch_add(1), u64::MAX);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);

    // SAFETY: Correct type.
    let _ = unsafe { Box::from_raw(dst.exit().0.as_ptr()) };
}

#[test]
fn test_and() {
    let dst = Box::new(0u64);
    let dst = unsafe { RacyMemory::enter_ptr(NonNull::from(Box::leak(dst))) };
    let dst_slice = dst.as_slice();
    let dst_u8 = dst_slice.as_aligned::<u8>().unwrap();
    let dst_u16 = dst_slice.as_aligned::<u16>().unwrap();
    let dst_u32 = dst_slice.as_aligned::<u32>().unwrap();
    let dst_u64 = dst_slice.as_aligned::<u64>().unwrap();

    assert_eq!(dst_u8.fetch_and(u8::MAX), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);
    dst_u64.store(u64::MAX, Ordering::Unordered);
    assert_eq!(dst_u8.fetch_and(u8::MAX), u8::MAX);
    assert_eq!(dst_u64.load(Ordering::Unordered), u64::MAX);
    assert_eq!(dst_u8.fetch_and(0xF0), u8::MAX);
    assert_eq!(dst_u64.load(Ordering::Unordered), u64::MAX - 0xF);
    assert_eq!(dst_u8.fetch_and(0), 0xF0);
    assert_eq!(dst_u64.load(Ordering::Unordered), u64::MAX - u8::MAX as u64);
    dst_u64.store(0, Ordering::Unordered);

    assert_eq!(dst_u16.fetch_and(u16::MAX), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);
    dst_u64.store(u64::MAX, Ordering::Unordered);
    assert_eq!(dst_u16.fetch_and(u16::MAX), u16::MAX);
    assert_eq!(dst_u64.load(Ordering::Unordered), u64::MAX);
    assert_eq!(dst_u16.fetch_and(0xFF00), u16::MAX);
    assert_eq!(dst_u64.load(Ordering::Unordered), u64::MAX - 0xFF);
    assert_eq!(dst_u16.fetch_and(0), 0xFF00);
    assert_eq!(
        dst_u64.load(Ordering::Unordered),
        u64::MAX - u16::MAX as u64
    );
    dst_u64.store(0, Ordering::Unordered);

    assert_eq!(dst_u32.fetch_and(u32::MAX), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);
    dst_u64.store(u64::MAX, Ordering::Unordered);
    assert_eq!(dst_u32.fetch_and(u32::MAX), u32::MAX);
    assert_eq!(dst_u64.load(Ordering::Unordered), u64::MAX);
    assert_eq!(dst_u32.fetch_and(0xFFFF_0000), u32::MAX);
    assert_eq!(dst_u64.load(Ordering::Unordered), u64::MAX - 0xFFFF);
    assert_eq!(dst_u32.fetch_and(0), 0xFFFF_0000);
    assert_eq!(
        dst_u64.load(Ordering::Unordered),
        u64::MAX - u32::MAX as u64
    );
    dst_u64.store(0, Ordering::Unordered);

    assert_eq!(dst_u64.fetch_and(u64::MAX), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);
    dst_u64.store(u64::MAX, Ordering::Unordered);
    assert_eq!(dst_u64.fetch_and(u64::MAX), u64::MAX);
    assert_eq!(dst_u64.load(Ordering::Unordered), u64::MAX);
    assert_eq!(dst_u64.fetch_and(0xFFFF_0000_FFFF_0000), u64::MAX);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xFFFF_0000_FFFF_0000);
    assert_eq!(dst_u64.fetch_and(0x0_FFFF_0000_FFFF), 0xFFFF_0000_FFFF_0000);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);

    // SAFETY: Correct type.
    let _ = unsafe { Box::from_raw(dst.exit().0.as_ptr()) };
}

#[test]
fn test_or() {
    let dst = Box::new(0u64);
    let dst = unsafe { RacyMemory::enter_ptr(NonNull::from(Box::leak(dst))) };
    let dst_slice = dst.as_slice();
    let dst_u8 = dst_slice.as_aligned::<u8>().unwrap();
    let dst_u16 = dst_slice.as_aligned::<u16>().unwrap();
    let dst_u32 = dst_slice.as_aligned::<u32>().unwrap();
    let dst_u64 = dst_slice.as_aligned::<u64>().unwrap();

    assert_eq!(dst_u8.fetch_or(0x73), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x73);
    assert_eq!(dst_u8.fetch_or(0x1B), 0x73);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x7B);
    assert_eq!(dst_u8.fetch_or(0xF0), 0x7B);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xFB);
    assert_eq!(dst_u8.fetch_or(0x00), 0xFB);
    assert_eq!(dst_u8.fetch_or(0xFF), 0xFB);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xFF);
    dst_u64.store(0, Ordering::Unordered);

    assert_eq!(dst_u16.fetch_or(0xB182), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xB182);
    assert_eq!(dst_u16.fetch_or(0x02C3), 0xB182);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xB3C3);
    assert_eq!(dst_u16.fetch_or(0xFF00), 0xB3C3);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xFFC3);
    assert_eq!(dst_u16.fetch_or(0), 0xFFC3);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xFFC3);
    assert_eq!(dst_u16.fetch_or(0x00FF), 0xFFC3);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xFFFF);
    assert_eq!(dst_u16.fetch_or(0), 0xFFFF);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xFFFF);
    dst_u64.store(0, Ordering::Unordered);

    assert_eq!(dst_u32.fetch_or(0x01A4_1005), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x01A4_1005);
    assert_eq!(dst_u32.fetch_or(0x5502_D581), 0x01A4_1005);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x55A6_D585);
    assert_eq!(dst_u32.fetch_or(0xFF00_FF00), 0x55A6_D585);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xFFA6_FF85);
    assert_eq!(dst_u32.fetch_or(0), 0xFFA6_FF85);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xFFA6_FF85);
    assert_eq!(dst_u32.fetch_or(0x00FF_00FF), 0xFFA6_FF85);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xFFFF_FFFF);
    dst_u64.store(0, Ordering::Unordered);

    assert_eq!(dst_u64.fetch_or(0xABCD_3456_01A4_1005), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xABCD_3456_01A4_1005);
    assert_eq!(
        dst_u64.fetch_or(0x0F25_0021_232B_C34A),
        0xABCD_3456_01A4_1005
    );
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xAFED_3477_23AF_D34F);
    assert_eq!(
        dst_u64.fetch_or(0xFF00_FF00_FF00_FF00),
        0xAFED_3477_23AF_D34F
    );
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xFFED_FF77_FFAF_FF4F);
    assert_eq!(dst_u64.fetch_or(0), 0xFFED_FF77_FFAF_FF4F);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xFFED_FF77_FFAF_FF4F);
    assert_eq!(
        dst_u64.fetch_or(0x00FF_00FF_00FF_00FF),
        0xFFED_FF77_FFAF_FF4F
    );
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xFFFF_FFFF_FFFF_FFFF);

    // SAFETY: Correct type.
    let _ = unsafe { Box::from_raw(dst.exit().0.as_ptr()) };
}

#[test]
fn test_xor() {
    let dst = Box::new(0u64);
    let dst = unsafe { RacyMemory::enter_ptr(NonNull::from(Box::leak(dst))) };
    let dst_slice = dst.as_slice();
    let dst_u8 = dst_slice.as_aligned::<u8>().unwrap();
    let dst_u16 = dst_slice.as_aligned::<u16>().unwrap();
    let dst_u32 = dst_slice.as_aligned::<u32>().unwrap();
    let dst_u64 = dst_slice.as_aligned::<u64>().unwrap();

    assert_eq!(dst_u8.fetch_xor(0x73), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x73);
    assert_eq!(dst_u8.fetch_xor(0x1B), 0x73);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x68);
    assert_eq!(dst_u8.fetch_xor(0xF0), 0x68);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x98);
    assert_eq!(dst_u8.fetch_xor(0x00), 0x98);
    assert_eq!(dst_u8.fetch_xor(0xFF), 0x98);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x67);
    assert_eq!(dst_u8.fetch_xor(0x67), 0x67);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);

    assert_eq!(dst_u16.fetch_xor(0xB182), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xB182);
    assert_eq!(dst_u16.fetch_xor(0x02C3), 0xB182);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xB341);
    assert_eq!(dst_u16.fetch_xor(0xFF00), 0xB341);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x4C41);
    assert_eq!(dst_u16.fetch_xor(0), 0x4C41);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x4C41);
    assert_eq!(dst_u16.fetch_xor(0x00FF), 0x4C41);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x4CBE);
    assert_eq!(dst_u16.fetch_xor(0xFFFF), 0x4CBE);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xB341);
    assert_eq!(dst_u16.fetch_xor(0xB341), 0xB341);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);

    assert_eq!(dst_u32.fetch_xor(0xA34B_B182), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xA34B_B182);
    assert_eq!(dst_u32.fetch_xor(0x86D0_02C3), 0xA34B_B182);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x259B_B341);
    assert_eq!(dst_u32.fetch_xor(0xFF00_FF00), 0x259B_B341);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xDA9B_4C41);
    assert_eq!(dst_u32.fetch_xor(0), 0xDA9B_4C41);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xDA9B_4C41);
    assert_eq!(dst_u32.fetch_xor(0x00FF_00FF), 0xDA9B_4C41);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xDA64_4CBE);
    assert_eq!(dst_u32.fetch_xor(0xFFFF_FFFF), 0xDA64_4CBE);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x259B_B341);
    assert_eq!(dst_u32.fetch_xor(0x259B_B341), 0x259B_B341);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);

    assert_eq!(dst_u64.fetch_xor(0x0567_98E0_A34B_B182), 0);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x0567_98E0_A34B_B182);
    assert_eq!(
        dst_u64.fetch_xor(0x1135_C732_86D0_02C3),
        0x0567_98E0_A34B_B182
    );
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x1452_5FD2_259B_B341);
    assert_eq!(
        dst_u64.fetch_xor(0xFF00_FF00_FF00_FF00),
        0x1452_5FD2_259B_B341
    );
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xEB52_A0D2_DA9B_4C41);
    assert_eq!(dst_u64.fetch_xor(0), 0xEB52_A0D2_DA9B_4C41);
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xEB52_A0D2_DA9B_4C41);
    assert_eq!(
        dst_u64.fetch_xor(0x00FF_00FF_00FF_00FF),
        0xEB52_A0D2_DA9B_4C41
    );
    assert_eq!(dst_u64.load(Ordering::Unordered), 0xEBAD_A02D_DA64_4CBE);
    assert_eq!(
        dst_u64.fetch_xor(0xFFFF_FFFF_FFFF_FFFF),
        0xEBAD_A02D_DA64_4CBE
    );
    assert_eq!(dst_u64.load(Ordering::Unordered), 0x1452_5FD2_259B_B341);
    assert_eq!(
        dst_u64.fetch_xor(0x1452_5FD2_259B_B341),
        0x1452_5FD2_259B_B341
    );
    assert_eq!(dst_u64.load(Ordering::Unordered), 0);

    // SAFETY: Correct type.
    let _ = unsafe { Box::from_raw(dst.exit().0.as_ptr()) };
}

#[test]
fn test_copy() {
    let mut src = Box::new([0u64; 16]);
    src[0] = u64::MAX;
    let src = unsafe { RacyMemory::enter_slice(NonNull::from(Box::leak(src))) };
    let src_slice = src.as_slice();
    let src_u8 = src_slice.to_bytes();
    let src_u16 = src_slice.align_to::<u16>().1;
    let src_u32 = src_slice.align_to::<u32>().1;
    let src_u64 = src_slice.align_to::<u64>().1;

    let dst = Box::new([0u64; 16]);
    let dst = unsafe { RacyMemory::enter_slice(NonNull::from(Box::leak(dst))) };
    let dst_slice = dst.as_slice();
    let dst_u8 = dst_slice.to_bytes();
    let dst_u16 = dst_slice.align_to::<u16>().1;
    let dst_u32 = dst_slice.align_to::<u32>().1;
    let dst_u64 = dst_slice.align_to::<u64>().1;

    dst_u8.slice_to(1).copy_from_racy_slice(&src_u8.slice_to(1));
    assert_eq!(dst.as_slice().load_unaligned::<u64>().unwrap(), 0xFF);
    dst_u16
        .slice_to(1)
        .copy_from_racy_slice(&src_u16.slice_to(1));
    assert_eq!(dst.as_slice().load_unaligned::<u64>().unwrap(), 0xFFFF);
    dst_u32
        .slice_to(1)
        .copy_from_racy_slice(&src_u32.slice_to(1));
    assert_eq!(dst.as_slice().load_unaligned::<u64>().unwrap(), 0xFFFF_FFFF);
    dst_u64
        .slice_to(1)
        .copy_from_racy_slice(&src_u64.slice_to(1));
    assert_eq!(
        dst.as_slice().load_unaligned::<u64>().unwrap(),
        0xFFFF_FFFF_FFFF_FFFF
    );
}
