# WS-Analysis
Analysis code for weather station data

This code has been used to produce the paper "Detailed Analysis of Local Climate at the CTAO-North Site from 20 Years of MAGIC Weather Station Data" 

1. Download the data from:  https://dx.doi.org/10.5281/zenodo.11279074 

2. Edit and run the python script create_files.py

3. Edit the __main__ part of the script WS.py to uncomment different parts of the analysis 

  #plot_datacount()
  #plot_downtime()
  #plot_temperature()
  #plot_DTR()
  #plot_temperature_not()
  #plot_snow()
  #plot_rainy_periods()
  #plot_humidity()
  #plot_humidity_not()
  #plot_pressure_not()
  #plot_pressure()    
  #plot_wind()
  #plot_huracans()

  As each individual part takes several minutes to finish and produces a long series of plots, I recommend to run WS.py by parts. 

  The script has been written for analysis of the MAGIC weather data only, a more user-friendly version easily adaptable for other weather station data is still being developed. 
  Contributions are welcome.
