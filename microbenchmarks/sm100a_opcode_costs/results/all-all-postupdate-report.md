# SM100A Opcode Cost Confirmation

- Total entries: 44
- Sync entries: 13
- Async entries: 31
- Direct evidence rows: 32
- Proxy rows: 8
- Unverified rows: 4
- Confirmed within +/-1 cycle(s): 40

## Mismatches

- `MOV`: expected 4, measured 0, delta -4 (unverified)

## Direct

- `FFMA`: measured 4 vs expected 4 (observed `FFMA` via `sync_ffma`)
- `FADD`: measured 4 vs expected 4 (observed `FADD` via `sync_fadd`)
- `FMUL`: measured 4 vs expected 4 (observed `FMUL` via `sync_fmul`)
- `HFMA2`: measured 4 vs expected 4 (observed `HFMA2` via `sync_hfma2`)
- `MUFU`: measured 17 vs expected 17 (observed `MUFU` via `sync_mufu`)
- `IMAD`: measured 4 vs expected 4 (observed `IMAD` via `sync_imad`)
- `IADD3`: measured 2 vs expected 2 (observed `IADD3` via `sync_iadd3`)
- `LOP3`: measured 4 vs expected 4 (observed `LOP3` via `sync_lop3`)
- `LDS`: measured 2 vs expected 2 (observed `LDS` via `sync_lds`)
- `SHFL`: measured 26 vs expected 26 (observed `SHFL` via `sync_shfl`)
- `S2R`: measured 2 vs expected 2 (observed `S2R` via `sync_s2r`)
- `UTCHMMA`: measured 54 vs expected 54 (observed `UTCHMMA` via `async_utchmma`)
- `UTMALDG`: measured 110 vs expected 110 (observed `UTMALDG` via `async_tma_load`)
- `UTMASTG`: measured 14 vs expected 14 (observed `UTMASTG` via `async_tma_store`)
- `UTMAREDG`: measured 45 vs expected 45 (observed `UTMAREDG` via `async_tma_reduce`)
- `UTMAPF`: measured 57 vs expected 57 (observed `UTMAPF` via `async_tma_prefetch`)
- `UTCCP`: measured 70 vs expected 70 (observed `UTCCP` via `async_utccp`)
- `LDT`: measured 0 vs expected 0 (observed `LDT` via `async_ldt`)
- `LDTM`: measured 0 vs expected 0 (observed `LDTM` via `async_ldtm`)
- `STT`: measured 2 vs expected 2 (observed `STT` via `async_stt`)
- `STTM`: measured 54 vs expected 54 (observed `STTM` via `async_sttm`)
- `LDGSTS`: measured 6 vs expected 6 (observed `LDGSTS` via `async_ldgsts`)
- `CCTL`: measured 4 vs expected 4 (observed `CCTL` via `async_cctl`)
- `UBLKCP`: measured 15 vs expected 15 (observed `UBLKCP` via `async_bulk_copy`)
- `UBLKPF`: measured 6 vs expected 6 (observed `UBLKPF` via `async_bulk_prefetch`)
- `UBLKRED`: measured 44 vs expected 44 (observed `UBLKRED` via `async_bulk_reduce`)
- `LDG`: measured 0 vs expected 0 (observed `LDG` via `async_ldg`)
- `STG`: measured 21 vs expected 21 (observed `STG` via `async_stg`)
- `LDL`: measured 0 vs expected 0 (observed `LDL` via `async_ldl`)
- `STL`: measured 4 vs expected 4 (observed `STL` via `async_stl`)
- `MBAR`: measured 7 vs expected 7 (observed `SYNCS.ARRIVE` via `async_mbar`)
- `BAR`: measured 20 vs expected 20 (observed `BAR` via `async_bar`)

## Proxy

- `STS`: measured 3 vs expected 3 (observed `STS` via `sync_sts`; {"target": 5.1894, "baseline": 2.4396, "proxy": false} | proxied via dependent STS+LDS chain)
- `UTCIMMA`: measured 54 vs expected 54 (observed `UTCHMMA` via `async_utchmma`; {"points": [[1.0, 52.75120849609377], [2.0, 105.75261230468752], [4.0, 213.25219726562503], [8.0, 431.7485717773437], [16.0, 855.7485717773437]], "intercept": -0.47692362467444127, "slope": 53.601218701434384} | proxied with f16 tcgen05.mma issue kernel)
- `UTCOMMA`: measured 54 vs expected 54 (observed `UTCHMMA` via `async_utchmma`; {"points": [[1.0, 52.75120849609377], [2.0, 105.75261230468752], [4.0, 213.25219726562503], [8.0, 431.7485717773437], [16.0, 855.7485717773437]], "intercept": -0.47692362467444127, "slope": 53.601218701434384} | proxied with f16 tcgen05.mma issue kernel)
- `UTCQMMA`: measured 54 vs expected 54 (observed `UTCHMMA` via `async_utchmma`; {"points": [[1.0, 52.75120849609377], [2.0, 105.75261230468752], [4.0, 213.25219726562503], [8.0, 431.7485717773437], [16.0, 855.7485717773437]], "intercept": -0.47692362467444127, "slope": 53.601218701434384} | proxied with f16 tcgen05.mma issue kernel)
- `UTMACCTL`: measured 57 vs expected 57 (observed `UTMAPF` via `async_tma_prefetch`; {"points": [[1.0, 56.277783203125], [2.0, 112.45999755859376], [4.0, 229.87843017578126], [8.0, 455.69129638671876], [16.0, 910.8273071289062]], "intercept": -0.1248260498046534, "slope": 56.959965958133814} | proxied with TMA prefetch/cache-control path)
- `UTCSHIFT`: measured 0 vs expected 0 (observed `LDTM` via `async_ldtm`; {"points": [[1.0, 0.4207885742187498], [2.0, 0.4069091796875002], [4.0, 0.4071655273437498], [8.0, 0.41022949218750004], [16.0, 0.4114624023437501]], "intercept": 0.41223347981770825, "slope": -0.00014878139700939773} | proxied with TMEM rearrangement/load path)
- `LDGMC`: measured 0 vs expected 0 (observed `LDG` via `async_ldg`; {"points": [[1.0, 1.49437255859375], [2.0, 0.05262451171875], [4.0, 0.04837646484375], [8.0, 0.04825439453125], [16.0, 0.04832763671875]], "intercept": 0.6524627685546874, "slope": -0.05065671859248991} | proxied with global load/cache-control path)
- `CCTLL`: measured 4 vs expected 4 (observed `CCTL` via `async_cctl`; {"points": [[1.0, 5.3094482421875], [2.0, 9.31156005859375], [4.0, 21.0484130859375], [8.0, 37.05322265625], [16.0, 69.04979248046875]], "intercept": 2.089505004882813, "slope": 4.236287467710434} | ptxas selected short-form cache-control encoding)

## Unverified

- `MOV`: no qualifying measurement+evidence pair
- `LDGDEPBAR`: no qualifying measurement+evidence pair
- `STAS`: no qualifying measurement+evidence pair
- `REDAS`: no qualifying measurement+evidence pair
