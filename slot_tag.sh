# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
# python3 test_slot.py --test_file "${1}" --pred_file "${2}"
python3 test_slot.py --test_file "./data/slot/test.json" --pred_file "./pred_slot.csv"