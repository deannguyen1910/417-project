This branch contains the code and instructions for partial expansion and lookahead. View other branches to see SMA* or learned policy.

The program is set to run the baseline A*, partial expansion, and lookahead using a partial k value of 3 and a lookahead depth of 4, a total node limit of 200000, and a timeout limit at 30, as set by parameters below.
It will run the 3 approaches for each test instance, recording the relevant results in results.csv.

To reproduce partial expansion and lookahead results:

First run: python run_experiments.py --instance "instances/test_*" --solver CBS --batch --run_all_modes --partial_k 3 --lookahead_depth 4 --node_limit 200000 --timeout 30

To generate analysis graphs and csvs:

Run: python analyze_results.py
