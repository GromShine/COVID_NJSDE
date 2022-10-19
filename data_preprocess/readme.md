## new data processes


pre_process.py/pre_process_for_all.py

1.1 read each county longitude and latitude and unique ID(fip)

1.2 calculate each county pair distance

1.3* Before, for every county, search its nearest 20 neighbors county to form a 21-dim data.
     Now, we find some county's nearest 20 neighbors may  have a distance more than 400 km.
      
1.4 the previous data is counted as an event occurrence with each new 500 cases, now 50.
