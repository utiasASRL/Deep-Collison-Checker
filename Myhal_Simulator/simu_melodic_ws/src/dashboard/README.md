## Desired dashboard functionality 
- Create a **Dashboard** object 
- Make **Series** objects with a set of query conditions
  - tour_name (string)
  - filter_status (boolean)
  - localization_technique (string)
  - success_status (boolean)
  - scenarios (list of strings)
  - earliest_time (date string: '%year-%month-%day-%hour-%minute-%second')
  - latest_time (date string)
- Create a **Display** object within the dashboard with the desired plots
  - types of plots:
    - text display listing the runs of a given Series (also printed to terminal)
    - translation_error plot
    - yaw_error plot
    - trajectory plot
    - path_diff plot
  - each plot will display the desired series (either average or showing all individual runs)
- Obtain stats about a given series in a dashboard 
  - average attributes (eg. localization error)
  - run names in a Series
- Visualize the bag file of a desired run
- Obtain meta-data about all runs
  - list of used localization techniques 
  - list of used tours *(TODO: list tour information)*
  - list of used scenarios *(TODO: list scenario information)*
  - number of runs that fall under a given query
  
```python
>>> from src.dashboard import * # TODO: add this to PYTHONSTARTUP env variable
>>> dashboard = Dashboard()
>>> series1 = Series(name = "series1", tour_name = 'A_tour', filter_status = 'true')
>>> series2 = Series(name = "series2" tour_name = 'A_tour', filter_status = 'false')
>>> dashboard.create_display(1,2) # one row and two colums of plots
>>> dashboard.add_series(series1, aggregate = True)
>>> dashboard.add_series(series2, aggregate = True)
>>> dashboard.plot_types([PlotTypes.text, PlotTypes.translation_error])
>>> dashboard.show()
>>> dashboard.list_runs("series1")
2020-06-25-20-50-16
2020-06-23-10-32-54
>>> dashboard.series_stats("series1", PlotTypes.translation_error)
average translation error: 0.05 m 
>>> dashboard.metadata_fields()
filter_status
localization_technique
scenarios
success_status
times
tour_names
>>> dashboard.metadata_fields('filter_status')
true
false
```