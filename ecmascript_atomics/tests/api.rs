// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use ecmascript_atomics::*;

#[test]
fn test_load() {
    let mut dst = Box::new(u64::MAX);
    let dst = std::slice::from_mut(dst.as_mut());

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .load(Ordering::Unordered),
        u8::MAX
    );
    assert_eq!(dst[0], u64::MAX);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .load(Ordering::Unordered),
        u16::MAX
    );
    assert_eq!(dst[0], u64::MAX);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .load(Ordering::Unordered),
        u32::MAX
    );
    assert_eq!(dst[0], u64::MAX);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .load(Ordering::Unordered),
        u64::MAX
    );
    assert_eq!(dst[0], u64::MAX);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .load(Ordering::SeqCst),
        u8::MAX
    );
    assert_eq!(dst[0], u64::MAX);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .load(Ordering::SeqCst),
        u16::MAX
    );
    assert_eq!(dst[0], u64::MAX);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .load(Ordering::SeqCst),
        u32::MAX
    );
    assert_eq!(dst[0], u64::MAX);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .load(Ordering::SeqCst),
        u64::MAX
    );
    assert_eq!(dst[0], u64::MAX);
}

#[test]
fn test_store() {
    let mut dst = Box::new(0u64);
    let dst = std::slice::from_mut(dst.as_mut());

    {
        let r = RacyMutSlice::from_mut_slice(dst);
        let r = r.as_slice().as_u8().unwrap();
        r.store(u8::MAX, Ordering::Unordered);
        assert_eq!(r.load(Ordering::Unordered), u8::MAX);
    }
    assert_eq!(dst[0], u8::MAX as u64);

    {
        let r = RacyMutSlice::from_mut_slice(dst);
        let r = r.as_slice().as_u16().unwrap();
        r.store(u16::MAX, Ordering::Unordered);
        assert_eq!(r.load(Ordering::Unordered), u16::MAX);
    }
    assert_eq!(dst[0], u16::MAX as u64);

    {
        let r = RacyMutSlice::from_mut_slice(dst);
        let r = r.as_slice().as_u32().unwrap();
        r.store(u32::MAX, Ordering::Unordered);
        assert_eq!(r.load(Ordering::Unordered), u32::MAX);
    }
    assert_eq!(dst[0], u32::MAX as u64);

    {
        let r = RacyMutSlice::from_mut_slice(dst);
        let r = r.as_slice().as_u64().unwrap();
        r.store(u64::MAX, Ordering::Unordered);
        assert_eq!(r.load(Ordering::Unordered), u64::MAX);
    }
    assert_eq!(dst[0], u64::MAX);
    dst[0] = 0;

    {
        let r = RacyMutSlice::from_mut_slice(dst);
        let r = r.as_slice().as_u8().unwrap();
        r.store(u8::MAX, Ordering::SeqCst);
        assert_eq!(r.load(Ordering::SeqCst), u8::MAX);
    }
    assert_eq!(dst[0], u8::MAX as u64);

    {
        let r = RacyMutSlice::from_mut_slice(dst);
        let r = r.as_slice().as_u16().unwrap();
        r.store(u16::MAX, Ordering::SeqCst);
        assert_eq!(r.load(Ordering::SeqCst), u16::MAX);
    }
    assert_eq!(dst[0], u16::MAX as u64);

    {
        let r = RacyMutSlice::from_mut_slice(dst);
        let r = r.as_slice().as_u32().unwrap();
        r.store(u32::MAX, Ordering::SeqCst);
        assert_eq!(r.load(Ordering::SeqCst), u32::MAX);
    }
    assert_eq!(dst[0], u32::MAX as u64);

    {
        let r = RacyMutSlice::from_mut_slice(dst);
        let r = r.as_slice().as_u64().unwrap();
        r.store(u64::MAX, Ordering::SeqCst);
        assert_eq!(r.load(Ordering::SeqCst), u64::MAX);
    }
    assert_eq!(dst[0], u64::MAX);
}

#[test]
fn test_exchange() {
    let mut dst = Box::new(0u64);
    let dst = std::slice::from_mut(dst.as_mut());

    {
        let r = RacyMutSlice::from_mut_slice(dst);
        let r = r.as_slice().as_u8().unwrap();
        assert_eq!(r.swap(u8::MAX), 0);
        assert_eq!(r.swap(0), u8::MAX);
    }
    assert_eq!(dst[0], 0);

    {
        let r = RacyMutSlice::from_mut_slice(dst);
        let r = r.as_slice().as_u16().unwrap();
        assert_eq!(r.swap(u16::MAX), 0);
        assert_eq!(r.swap(0), u16::MAX);
    }
    assert_eq!(dst[0], 0);

    {
        let r = RacyMutSlice::from_mut_slice(dst);
        let r = r.as_slice().as_u32().unwrap();
        assert_eq!(r.swap(u32::MAX), 0);
        assert_eq!(r.swap(0), u32::MAX);
    }
    assert_eq!(dst[0], 0);

    {
        let r = RacyMutSlice::from_mut_slice(dst);
        let r = r.as_slice().as_u64().unwrap();
        assert_eq!(r.swap(u64::MAX), 0);
        assert_eq!(r.swap(0), u64::MAX);
    }
    assert_eq!(dst[0], 0);
}

#[test]
fn test_compare_exchange() {
    let mut dst = Box::new(0u64);
    let dst = std::slice::from_mut(dst.as_mut());

    {
        assert_eq!(
            RacyMutSlice::from_mut_slice(dst)
                .as_slice()
                .as_u8()
                .unwrap()
                .compare_exchange(u8::MAX, u8::MAX),
            Err(0)
        );
    }
    assert_eq!(dst[0], 0);
    {
        assert_eq!(
            RacyMutSlice::from_mut_slice(dst)
                .as_slice()
                .as_u8()
                .unwrap()
                .compare_exchange(0, u8::MAX),
            Ok(0)
        );
    }
    assert_eq!(dst[0], u8::MAX as u64);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .compare_exchange(0, 0),
        Err(u8::MAX)
    );
    assert_eq!(dst[0], u8::MAX as u64);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .compare_exchange(u8::MAX, 0),
        Ok(u8::MAX)
    );
    assert_eq!(dst[0], 0);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .compare_exchange(u16::MAX, u16::MAX),
        Err(0)
    );
    assert_eq!(dst[0], 0);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .compare_exchange(0, u16::MAX),
        Ok(0)
    );
    assert_eq!(dst[0], u16::MAX as u64);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .compare_exchange(0, 0),
        Err(u16::MAX)
    );
    assert_eq!(dst[0], u16::MAX as u64);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .compare_exchange(u16::MAX, 0),
        Ok(u16::MAX)
    );
    assert_eq!(dst[0], 0);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .compare_exchange(u32::MAX, u32::MAX),
        Err(0)
    );
    assert_eq!(dst[0], 0);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .compare_exchange(0, u32::MAX),
        Ok(0)
    );
    assert_eq!(dst[0], u32::MAX as u64);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .compare_exchange(0, 0),
        Err(u32::MAX)
    );
    assert_eq!(dst[0], u32::MAX as u64);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .compare_exchange(u32::MAX, 0),
        Ok(u32::MAX)
    );
    assert_eq!(dst[0], 0);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .compare_exchange(u64::MAX, u64::MAX),
        Err(0)
    );
    assert_eq!(dst[0], 0);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .compare_exchange(0, u64::MAX),
        Ok(0)
    );
    assert_eq!(dst[0], u64::MAX as u64);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .compare_exchange(0, 0),
        Err(u64::MAX)
    );
    assert_eq!(dst[0], u64::MAX as u64);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .compare_exchange(u64::MAX, 0),
        Ok(u64::MAX)
    );
    assert_eq!(dst[0], 0);
}

#[test]
fn test_add() {
    let mut dst = Box::new(0u64);
    let dst = std::slice::from_mut(dst.as_mut());

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_add(u8::MAX),
        0
    );
    assert_eq!(dst[0], u8::MAX as u64);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_add(1),
        u8::MAX
    );
    assert_eq!(dst[0], 0);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_add(u16::MAX),
        0
    );
    assert_eq!(dst[0], u16::MAX as u64);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_add(1),
        u16::MAX
    );
    assert_eq!(dst[0], 0);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_add(u32::MAX),
        0
    );
    assert_eq!(dst[0], u32::MAX as u64);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_add(1),
        u32::MAX
    );
    assert_eq!(dst[0], 0);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_add(u64::MAX),
        0
    );
    assert_eq!(dst[0], u64::MAX);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_add(1),
        u64::MAX
    );
    assert_eq!(dst[0], 0);
}

#[test]
fn test_and() {
    let mut dst = Box::new(0u64);
    let dst = std::slice::from_mut(dst.as_mut());

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_and(u8::MAX),
        0
    );
    assert_eq!(dst[0], 0);
    dst[0] = u64::MAX;
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_and(u8::MAX),
        u8::MAX
    );
    assert_eq!(dst[0], u64::MAX);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_and(0xF0),
        u8::MAX
    );
    assert_eq!(dst[0], u64::MAX - 0xF);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_and(0),
        0xF0
    );
    assert_eq!(dst[0], u64::MAX - u8::MAX as u64);
    dst[0] = 0;

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_and(u16::MAX),
        0
    );
    assert_eq!(dst[0], 0);
    dst[0] = u64::MAX;
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_and(u16::MAX),
        u16::MAX
    );
    assert_eq!(dst[0], u64::MAX);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_and(0xFF00),
        u16::MAX
    );
    assert_eq!(dst[0], u64::MAX - 0xFF);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_and(0),
        0xFF00
    );
    assert_eq!(dst[0], u64::MAX - u16::MAX as u64);
    dst[0] = 0;

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_and(u32::MAX),
        0
    );
    assert_eq!(dst[0], 0);
    dst[0] = u64::MAX;
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_and(u32::MAX),
        u32::MAX
    );
    assert_eq!(dst[0], u64::MAX);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_and(0xFFFF_0000),
        u32::MAX
    );
    assert_eq!(dst[0], u64::MAX - 0xFFFF);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_and(0),
        0xFFFF_0000
    );
    assert_eq!(dst[0], u64::MAX - u32::MAX as u64);
    dst[0] = 0;

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_and(u64::MAX),
        0
    );
    assert_eq!(dst[0], 0);
    dst[0] = u64::MAX;
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_and(u64::MAX),
        u64::MAX
    );
    assert_eq!(dst[0], u64::MAX);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_and(0xFFFF_0000_FFFF_0000),
        u64::MAX
    );
    assert_eq!(dst[0], 0xFFFF_0000_FFFF_0000);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_and(0x0_FFFF_0000_FFFF),
        0xFFFF_0000_FFFF_0000
    );
    assert_eq!(dst[0], 0);
}

#[test]
fn test_or() {
    let mut dst = Box::new(0u64);
    let dst = std::slice::from_mut(dst.as_mut());

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_or(0x73),
        0
    );
    assert_eq!(dst[0], 0x73);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_or(0x1B),
        0x73
    );
    assert_eq!(dst[0], 0x7B);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_or(0xF0),
        0x7B
    );
    assert_eq!(dst[0], 0xFB);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_or(0x00),
        0xFB
    );
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_or(0xFF),
        0xFB
    );
    assert_eq!(dst[0], 0xFF);
    dst[0] = 0;

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_or(0xB182),
        0
    );
    assert_eq!(dst[0], 0xB182);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_or(0x02C3),
        0xB182
    );
    assert_eq!(dst[0], 0xB3C3);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_or(0xFF00),
        0xB3C3
    );
    assert_eq!(dst[0], 0xFFC3);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_or(0),
        0xFFC3
    );
    assert_eq!(dst[0], 0xFFC3);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_or(0x00FF),
        0xFFC3
    );
    assert_eq!(dst[0], 0xFFFF);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_or(0),
        0xFFFF
    );
    assert_eq!(dst[0], 0xFFFF);
    dst[0] = 0;

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_or(0x01A4_1005),
        0
    );
    assert_eq!(dst[0], 0x01A4_1005);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_or(0x5502_D581),
        0x01A4_1005
    );
    assert_eq!(dst[0], 0x55A6_D585);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_or(0xFF00_FF00),
        0x55A6_D585
    );
    assert_eq!(dst[0], 0xFFA6_FF85);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_or(0),
        0xFFA6_FF85
    );
    assert_eq!(dst[0], 0xFFA6_FF85);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_or(0x00FF_00FF),
        0xFFA6_FF85
    );
    assert_eq!(dst[0], 0xFFFF_FFFF);
    dst[0] = 0;

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_or(0xABCD_3456_01A4_1005),
        0
    );
    assert_eq!(dst[0], 0xABCD_3456_01A4_1005);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_or(0x0F25_0021_232B_C34A),
        0xABCD_3456_01A4_1005
    );
    assert_eq!(dst[0], 0xAFED_3477_23AF_D34F);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_or(0xFF00_FF00_FF00_FF00),
        0xAFED_3477_23AF_D34F
    );
    assert_eq!(dst[0], 0xFFED_FF77_FFAF_FF4F);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_or(0),
        0xFFED_FF77_FFAF_FF4F
    );
    assert_eq!(dst[0], 0xFFED_FF77_FFAF_FF4F);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_or(0x00FF_00FF_00FF_00FF),
        0xFFED_FF77_FFAF_FF4F
    );
    assert_eq!(dst[0], 0xFFFF_FFFF_FFFF_FFFF);
}

#[test]
fn test_xor() {
    let mut dst = Box::new(0u64);
    let dst = std::slice::from_mut(dst.as_mut());

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_xor(0x73),
        0
    );
    assert_eq!(dst[0], 0x73);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_xor(0x1B),
        0x73
    );
    assert_eq!(dst[0], 0x68);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_xor(0xF0),
        0x68
    );
    assert_eq!(dst[0], 0x98);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_xor(0x00),
        0x98
    );
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_xor(0xFF),
        0x98
    );
    assert_eq!(dst[0], 0x67);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u8()
            .unwrap()
            .fetch_xor(0x67),
        0x67
    );
    assert_eq!(dst[0], 0);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_xor(0xB182),
        0
    );
    assert_eq!(dst[0], 0xB182);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_xor(0x02C3),
        0xB182
    );
    assert_eq!(dst[0], 0xB341);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_xor(0xFF00),
        0xB341
    );
    assert_eq!(dst[0], 0x4C41);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_xor(0),
        0x4C41
    );
    assert_eq!(dst[0], 0x4C41);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_xor(0x00FF),
        0x4C41
    );
    assert_eq!(dst[0], 0x4CBE);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_xor(0xFFFF),
        0x4CBE
    );
    assert_eq!(dst[0], 0xB341);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u16()
            .unwrap()
            .fetch_xor(0xB341),
        0xB341
    );
    assert_eq!(dst[0], 0);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_xor(0xA34B_B182),
        0
    );
    assert_eq!(dst[0], 0xA34B_B182);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_xor(0x86D0_02C3),
        0xA34B_B182
    );
    assert_eq!(dst[0], 0x259B_B341);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_xor(0xFF00_FF00),
        0x259B_B341
    );
    assert_eq!(dst[0], 0xDA9B_4C41);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_xor(0),
        0xDA9B_4C41
    );
    assert_eq!(dst[0], 0xDA9B_4C41);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_xor(0x00FF_00FF),
        0xDA9B_4C41
    );
    assert_eq!(dst[0], 0xDA64_4CBE);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_xor(0xFFFF_FFFF),
        0xDA64_4CBE
    );
    assert_eq!(dst[0], 0x259B_B341);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u32()
            .unwrap()
            .fetch_xor(0x259B_B341),
        0x259B_B341
    );
    assert_eq!(dst[0], 0);

    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_xor(0x0567_98E0_A34B_B182),
        0
    );
    assert_eq!(dst[0], 0x0567_98E0_A34B_B182);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_xor(0x1135_C732_86D0_02C3),
        0x0567_98E0_A34B_B182
    );
    assert_eq!(dst[0], 0x1452_5FD2_259B_B341);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_xor(0xFF00_FF00_FF00_FF00),
        0x1452_5FD2_259B_B341
    );
    assert_eq!(dst[0], 0xEB52_A0D2_DA9B_4C41);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_xor(0),
        0xEB52_A0D2_DA9B_4C41
    );
    assert_eq!(dst[0], 0xEB52_A0D2_DA9B_4C41);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_xor(0x00FF_00FF_00FF_00FF),
        0xEB52_A0D2_DA9B_4C41
    );
    assert_eq!(dst[0], 0xEBAD_A02D_DA64_4CBE);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_xor(0xFFFF_FFFF_FFFF_FFFF),
        0xEBAD_A02D_DA64_4CBE
    );
    assert_eq!(dst[0], 0x1452_5FD2_259B_B341);
    assert_eq!(
        RacyMutSlice::from_mut_slice(dst)
            .as_slice()
            .as_u64()
            .unwrap()
            .fetch_xor(0x1452_5FD2_259B_B341),
        0x1452_5FD2_259B_B341
    );
    assert_eq!(dst[0], 0);
}

#[test]
fn test_copy() {
    let mut src = Box::new([0u64; 16]);
    let mut dst = Box::new([0u64; 16]);
    src[0] = u64::MAX;

    {
        let s = RacyMutSlice::from_mut_slice(src.as_mut_slice());
        let s = s.as_slice();

        assert_eq!(s.align_to::<u8>().1.slice_to(1).len(), 1);
        RacyMutSlice::from_mut_slice(dst.as_mut_slice())
            .as_slice()
            .align_to::<u8>()
            .1
            .slice_to(1)
            .copy_from_slice(&s.align_to::<u8>().1.slice_to(1));
        assert_eq!(dst[0], 0xFF);
        RacyMutSlice::from_mut_slice(dst.as_mut_slice())
            .as_slice()
            .align_to::<u16>()
            .1
            .slice_to(1)
            .copy_from_slice(&s.align_to::<u16>().1.slice_to(1));
        assert_eq!(dst[0], 0xFFFF);
        RacyMutSlice::from_mut_slice(dst.as_mut_slice())
            .as_slice()
            .align_to::<u32>()
            .1
            .slice_to(1)
            .copy_from_slice(&s.align_to::<u32>().1.slice_to(1));
        assert_eq!(dst[0], 0xFFFF_FFFF);
        RacyMutSlice::from_mut_slice(dst.as_mut_slice())
            .as_slice()
            .align_to::<u64>()
            .1
            .slice_to(1)
            .copy_from_slice(&s.align_to::<u64>().1.slice_to(1));
        assert_eq!(dst[0], 0xFFFF_FFFF_FFFF_FFFF);
    }
}
