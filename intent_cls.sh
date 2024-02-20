# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
# python3 test_intent.py --test_file "${1}" --pred_file "${2}"
python3 test_intent.py --test_file "./data/intent/test.json" --pred_file "./pred_intent.csv" 