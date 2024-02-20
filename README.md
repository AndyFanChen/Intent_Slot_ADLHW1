# 2022 ADLHW1

## Step1 download
this code will download the model from dropbox.
```
bash download.sh
```

## Step2 unzip and open the floder
After download and unzip the floder you can run this.
```
unzip r10723050ADLHW1.zip?dl=1
cd "r10723050 ADLHW1/src"
```
## Step3 run the code
There are two python file for Q1: intent classification problem  and Q2: slot tagging problem
### Q1: intent classification problem
run in this format:
```
python3 test_intent.py --test_file "${1}" --pred_file "${2}"
```
If using the data contained in download.sh，you can run by: 
```
python3 test_intent.py --test_file src\data\intent\test.json --pred_file src\pred_intent.csv 
```




### Q2: intent classification problem
run in this format:
```
python3 src/test_slot.py --test_file "${1}" --pred_file "${2}"
```
If using the data contained in download.sh，you can run by: 
```
python3 src/test_slot.py --test_file src\data\slot\test.json --pred_file src\pred_slot.csv 
```

