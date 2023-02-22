# Generating verilog basic testbench format automatically.
1. Use the ctr+shift+p command to select tesetbench then with python3 testbench can be quickly generated.
2. However, problem occurs since python3 cannot find the file,that is because it serach the file using "\", but for it to work we must search in "/", so just change every "\" into "/"