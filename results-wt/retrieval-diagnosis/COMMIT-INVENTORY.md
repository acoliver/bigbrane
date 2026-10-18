# Small checkpoint inventory

These explicit paths are suitable for a future checkpoint on sibling r2-bar. Nothing is staged or committed here. Runtime results are locally ignored, so select report/tool paths explicitly instead of adding the entire results tree. Existing research/host-reference work is excluded.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `hybrid/literature/sources/arxiv.py` | 3205 | `be20d9f3f0ceb52b2b2317ea0f18f4256f031c171390ffc3374a9411f6d61c11` |
| `hybrid/tests/literature/test_parsing.py` | 5438 | `57e7ea64ffed7fa9f565878bfd1d1945eb19297719ae5c2414c6e4aaa8b1e212` |
| `hybrid/tests/literature/test_arxiv_https.py` | 1520 | `97136a8cc659e1f1ce20b2680461dbaad96887d210c4ab2540cc440fd68eaa5c` |
| `results-wt/retrieval-diagnosis/diagnose.py` | 18368 | `d80f49e4fae115763fafe04fb1ba6d38b418ba7a0f5d3baf9c805f256702def2` |
| `results-wt/retrieval-diagnosis/audit_raw.py` | 8406 | `fe51bc4f175bfd033f9e6e8bdf6061dda4cedb00b875548ccb69e323126f394d` |
| `results-wt/retrieval-diagnosis/summarize.py` | 6405 | `a3570d6056bd1b269a1c5c7fb1914d809d877a738f708d3758816801aca82676` |
| `results-wt/retrieval-diagnosis/validate_transport.py` | 2693 | `844dcf647888d98abb63b7ea24c39b5e08404a2e343c6e778bf21f7031e00b12` |
| `results-wt/retrieval-diagnosis/test_diagnose.py` | 7877 | `212d30175b1b41275525f5294bd0e733a357d6265dc0ccb1be2bbf579f95e301` |
| `results-wt/retrieval-diagnosis/REPORT.md` | 29553 | `727fa5a91f42674ab69966e9d5e7b924749a3427395852d050e011fb5785674f` |
| `results-wt/retrieval-diagnosis/QUERIES.md` | 26074 | `5ec2d0d4a1bd1be9f6eb1179c50ba2ac5ef703032b9331cb6b46aef0f17da686` |
| `results-wt/retrieval-diagnosis/STATUS.md` | 2822 | `7fd24f9a190f75c4785071c42050c33fa1a22e7fa66a2bb4a0dd8d483a0afa3c` |
| `results-wt/retrieval-diagnosis/PLACEMENT.md` | 8039 | `206e3694d0a539b751bbed7ab0cb2bb7fa28297914c8a0bb46661e3cb952f6bd` |
| `results-wt/retrieval-diagnosis/PLACEMENT-BEGIN.json` | 449 | `dfe88dba38d4e4ccf32ebd4eadc7f7fb20756d0075c4a112d9d098ef91e63971` |
| `results-wt/retrieval-diagnosis/PLACEMENT-COPY.json` | 13369 | `148b2f7ef6fd68792a0904c3eb894ad5d5401e876a58b52a7825977afc83c753` |
| `results-wt/retrieval-diagnosis/PLACEMENT-VERIFY.json` | 16194 | `9f5ecb5cd1b5eb5ebe3208ac4eea159c5ad8e8a766048ec0987fff3ed0601b4b` |
| `results-wt/retrieval-diagnosis/placement-originals/arxiv-main.patch` | 578 | `0399e817803362033fd45fcd63641605feedea56c4ea7782968d2875c6a69232` |
| `results-wt/retrieval-diagnosis/placement-originals/test_arxiv_https.py` | 1520 | `97136a8cc659e1f1ce20b2680461dbaad96887d210c4ab2540cc440fd68eaa5c` |
| `results-wt/retrieval-diagnosis/placement-verification/tests.json` | 1735 | `ac69f3c7e655572e30e2c4a84e73de603bac257e761ac111f3d4b45eb2a02d6e` |
| `results-wt/retrieval-diagnosis/placement-verification/offline-replay.json` | 473 | `13946548adc708d64f0fadba8c6bebbe47b5d6ad008454568f358a31e20d3ad7` |

The inventory itself (`results-wt/retrieval-diagnosis/COMMIT-INVENTORY.md`) may accompany these paths. FILEMANIFEST.json remains a local full-tree receipt, outside this small checkpoint. Exclude http.jsonl, postfix-http.jsonl, diagnostic-cache/, postfix-cache/, placement-verification/pytest-*/, all logs, offline.json, raw-audit.json, MEASUREMENTS.json, *-plan*.json, fetched HTML/PDF files and bytecode. Those files remain preserved locally and hashed. Do not stage the entire directory.

PLACEMENT-COPY.json hashes every original diagnostic artifact. PLACEMENT-VERIFY.json supplies current source/test/tool hashes, clean-main receipt, concurrent-worker observations and exact verification commands. The preserved patch and test retain the misplaced change without including historical output datasets.
