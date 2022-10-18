## small example data use 'm1' : Massachusetts from 20200928 to 20201228
## big example data use 'c1': california from 20200928 to 20201228

## new data processes


pre_process.py/pre_process_for_all.py

1.1 read each county longitude and latitude and unique ID(fip)

1.2 calculate each county pair distance

1.3 for every county, search its nearest 20 neighbors county to form a 21-dim data.

1.4 the previous data is counted as an event occurrence with each new 500 cases, now 50.
