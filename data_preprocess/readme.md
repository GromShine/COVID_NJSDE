## new data processes


pre_process_ver_num.py

Given a county ID, start time, end time, and the number of nearby counties $N$, output a dataset of the $N$ nearest neighbor counties of the target county.

pre_process_ver_dis.py

Given a county ID, start time, end time, and the distance, ouput an covid cases dataset centered on that county within the distance

extract_top2.py

For meta-learning, we pick up 2 county in each state, whose neighborhood with the least empty sequence and the most abundant data.
