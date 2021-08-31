### Background Information
What is geothermal energy? Geothermal is the natural heat of the Earth derived from the decay of the radioactive elements in the Earth’s crust and transferred to the subsurface by conduction and convection. 
Temperatures at the core–mantle boundary may reach over 4000 °C (7200 °F). The high temperature and pressure in Earth's interior cause some rock to melt and solid mantle to behave plastically, resulting in parts of the mantle convecting upward since it is lighter than the surrounding rock. Rock and water is heated in the crust, sometimes up to 370 °C (700 °F).
What is geothermal used for? For centuries, geothermal springs have been used for bathing, heating and cooking. But in the early 20th century people started to consider geothermal as a practical source of energy with huge potential. Geothermal energy is now used to produce electricity, to heat and cool buildings as well as for other industrial purposes like fruit and vegetable cultivation. 
Reference: https://en.wikipedia.org/wiki/Geothermal_energy, https://www.geothermal-energy.org/explore/what-is-geothermal/


**Problem Statement**
2 parameters are important in the evaluation of geothermal potential. 
Formation Temperature: The higher the temperature at bottom, the higher the potential for Geothermal use. Despite the abundance of techniques for collecting drilling and well operation data, they do not necessarily provide the real Bottom Hole Temperature (BHT). 
Flowrate Capacity: Flowrate determines the amount of fluid that can flow naturally or be pumped through the formation or pipe. Higher permeability formations or larger diameter pipe are more viable for Geothermal purposes.

There is the belief that old (or current) oil and gas wells can be re-purposed for geothermal energy use, using the subsurface infrastructure already in place plus the data already collected during the life of the well.

We are provided with relevant well data from 2 Oil & Gas basins: 
-	Duvernay in Alberta, Canada
-	Eaglebine in Texas, USA 

The information includes;
-	Well logs in LAS format
-	Well header, drilling, completion and production data
-	BHT and DST Temperature measurements
-	Calculated synthetic true temperature

The goal of this project is to evaluate what wells are showing geothermal potential and based on those, which areas deserve further evaluation.  The goal of this notebook is to develop a machine learning model to predict the real bottom hole temperature and determine a subset of available wells showing the most potential for geothermal energy use.  




We are provided with relevant well data from 2 Oil & Gas basins: 
•	Duvernay in Alberta, Canada
•	Eaglebine in Texas, USA 

The information includes;
•	Well logs in LAS format
•	Well header, drilling, completion and production data
•	BHT and DST Temperature measurements
•	Calculated synthetic true temperature

The goal of this project is to evaluate what wells are showing geothermal potential and based on those, which areas deserve further evaluation.  The goal of this notebook is to develop a machine learning model to predict the real bottom hole temperature and determine a subset of available wells showing the most potential for geothermal energy use.  

### The Dataset 
We were provided with relevant oil and gas well data from two basins:  Duvernay in Alberta, Canada and Eaglebine in Texas, USA. An overview of the data is provided below: 
  * **True Temperature Train:** True Temperature data at multiple depths provided by data vendors derived using their proprietary methods. The prediction output (y variable) is the true temperature at bottom hole depth of the well. 
  * **Static Temperature logs:** Actual static formation temperatures for some wells recorded at bottom hole. In cases where this is available, this will be the prediction output (y variable). 
  * **Formation Tops:** Geologic formations and their Subsea depth for a particular well.  
  * **Well Headers:** Meta data related to the most important parameters of the wells including surface/bottom hole Latitude/Longitude, elevation, and total depths (TD). 
  * **Production Summary:** Production parameters related to a well including first production dates, total and maximum production (Oil/Gas/Water). 
  * **Mud Image Log:** Mud log files in TIF digital format. 
  * **Well Log files:** Digitalized well logs for each well in LAS format. Common well logs in each file are Gamma ray, Neutron Porosity, Density, etc.  
  * **DST Temperature and Pressure (Duvernay only):** Temperature and pressure measurements taken with Drill Stem Tests (DST) while drilling. This includes metadata related to the DST.  
  * **BHT TSC (Eaglebine only):** Bottom hole temperatures (BHT) plus Time Since Circulation information (TSC, elapsed time since last circulation before temperature was measured). 
  * **Casing & Production Summary (Eaglebine only):** Minimum casing size, completion and spud dates and cumulative oil/gas/water volumes. 
  * **Mud Weights (Eaglebine only):** Depths and mud weights in pounds per gallon (ppg). 

<details>
  <summary>Click to expand full data dictionary</summary>

Number | Basin | File name | Field name | Definition | Description
---- | ---- | ---- | ---- | ----
1 | Duverney | Duvernay DST Pressures SPE May 2 2021 | Well ID | Unique ID of well
2 | Duverney | Duvernay DST Pressures SPE May 2 2021 | KB Elev (m) | Kelly Bushing elevation above reference datum (ground or mean sea level)
3 | Duverney | Duvernay DST Pressures SPE May 2 2021 | DST Number | Drill stem testing (DST) sequence number per formation
4 | Duverney | Duvernay DST Pressures SPE May 2 2021 | Formation DSTd | Formation DST was performed in
5 | Duverney | Duvernay DST Pressures SPE May 2 2021 | DST Start Depth (TVD) (m) | Top depth of DST section (TVD)
6 | Duverney | Duvernay DST Pressures SPE May 2 2021 | DST End Depth (TVD) (m) | Bottom depth of DST section (TVD)
7 | Duverney | Duvernay DST Pressures SPE May 2 2021 | DST Start Depth (MD) (m) | Top depth of DST section (measured depth)
8 | Duverney | Duvernay DST Pressures SPE May 2 2021 | DST End Depth (MD) (m) | Bottom depth of DST section (measured depth)
9 | Duverney | Duvernay DST Pressures SPE May 2 2021 | DST Test Date | Date DST test performed
10 | Duverney | Duvernay DST Pressures SPE May 2 2021 | Test Type | Type of DST test performed (DST, WLT, LRT)
11 | Duverney | Duvernay DST Pressures SPE May 2 2021 | DST Misrun | DST failure (Y or N)
12 | Duverney | Duvernay DST Pressures SPE May 2 2021 | Misrun Problem Type | DST failure type
13 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 1st Valve Open Time | Time in minutes valve is opened on first test
14 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 2nd Valve Open Time | Time in minutes valve is opened on second test
15 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 3rd Valve Open Time | Time in minutes valve is opened on third test
16 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 1st Shut-in Time | Time in minutes valve is shut in on first test
17 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 2nd Shut-in Time | Time in minutes valve is shut in on second test
18 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 3rd Shut-in Time | Time in minutes valve is shut in on third test
19 | Duverney | Duvernay DST Pressures SPE May 2 2021 | Pressure Recorder Depth (m) | Depth of pressure sensor relative to datum
20 | Duverney | Duvernay DST Pressures SPE May 2 2021 | DST Bottom Hole Temp. (degC) | Bottom hole temperature recorded on DST tool
21 | Duverney | Duvernay DST Pressures SPE May 2 2021 | Initial Hydrostatic Pressure (kPa) | Initial Hydrostatic Pressure at start of DST
22 | Duverney | Duvernay DST Pressures SPE May 2 2021 | Final Hydrostatic Pressure (kPa) | Initial Hydrostatic Pressure at end of DST
23 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 1st Flow Pressure (kPa) | Flowing pressure on first valve open test
24 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 2nd Flow Pressure (kPa) | Flowing pressure on second valve open test
25 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 3rd Flow Pressure (kPa) | Flowing pressure on third valve open test
26 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 1st Shut-in Pressure (kPa) | Shut in pressure on first shut in test
27 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 1st Shut-in Initial Slope | Shut in pressure on second valve shut in test
28 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 1st Shut-in Final Slope | Shut in pressure on third valve shut in test
29 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 1st Shut-in Extrapolated Press (kPa) | Extrapolated pressure from measurements in first shut in test
30 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 2nd Shut-in Pressure (kPa) | Shut in pressure on second valve shut in test
31 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 2nd Shut-in Initial Slope | Initial recorded pressure slope of second shut in test
32 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 2nd Shut-in Final Slope | Final recorded pressure slope of second shut in test
33 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 2nd Shut-in Extrapolated Press (kPa) | Extrapolated pressure from measurements in second shut in test
34 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 3rd Shut-in Pressure (kPa) | Shut in pressure on third valve shut in test
35 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 3rd Shut-in Initial Slope | Initial recorded pressure slope of third shut in test
36 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 3rd Shut-in Final Slope | Final recorded pressure slope of third shut in test
37 | Duverney | Duvernay DST Pressures SPE May 2 2021 | 3rd Shut-in Extrapolated Press (kPa) | Extrapolated pressure from measurements in third shut in test
38 | Duverney | Duvernay DST Pressures SPE May 2 2021 | Maximum Shut-in Pressure (kPa) | Maximum recorded pressure from DST shut in tests
42 | Duverney | Duvernay DST BHT for SPE April 20 2021 | Well ID | Unique ID of well
43 | Duverney | Duvernay DST BHT for SPE April 20 2021 | DST Start Depth (MD) (m) | Top depth of DST section (measured depth)
44 | Duverney | Duvernay DST BHT for SPE April 20 2021 | DST End Depth (MD) (m) | Bottom depth of DST section (measured depth)
45 | Duverney | Duvernay DST BHT for SPE April 20 2021 | DST Bottom Hole Temp. (degC) | Bottom hole temperature recorded on DST tool
46 | Duverney | Duvernay DST BHT for SPE April 20 2021 | DST Test Date | Date DST test performed
47 | Duverney | Duvernay DST BHT for SPE April 20 2021 | Test Type | Type of DST test performed (DST, WLT, LRT)
48 | Duverney | Duvernay DST BHT for SPE April 20 2021 | DST Misrun | DST failure (Y or N)
49 | Duverney | Duvernay DST BHT for SPE April 20 2021 | DST Number | DST sequence number per formation
50 | Duverney | Duvernay DST BHT for SPE April 20 2021 | Formation DSTd | Formation DST was performed in
51 | Duverney | Duvernay DST BHT for SPE April 20 2021 | elevation M above sea level | Elevation measurement above mean sea level
52 | Duverney | Duvernay DST BHT for SPE April 20 2021 | UWI | Unique well Identifier
53 | Duverney | Duvernay formation tops SPE April 20 2021 | UWI | Unique well Identifier
54 | Duverney | Duvernay formation tops SPE April 20 2021 | Bottom Hole Location X_m_NAD27_Zone 11N (120 W to 114 W) | X coordinates
55 | Duverney | Duvernay formation tops SPE April 20 2021 | Bottom Hole Location Y_m_NAD27_Zone 11N (120 W to 114 W) | Y coordinates
56 | Duverney | Duvernay formation tops SPE April 20 2021 | Elevation(m above sea level) | Elevation measurement above mean sea level
57 | Duverney | Duvernay formation tops SPE April 20 2021 | 01_Battle (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
58 | Duverney | Duvernay formation tops SPE April 20 2021 | 02_Lea_Park (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
59 | Duverney | Duvernay formation tops SPE April 20 2021 | 03_1st_White_Speckled_Shale (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
60 | Duverney | Duvernay formation tops SPE April 20 2021 | 04_2nd_White_Speckled_Shale (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
61 | Duverney | Duvernay formation tops SPE April 20 2021 | 05_Fish_scales (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
62 | Duverney | Duvernay formation tops SPE April 20 2021 | 06_Mannville_Top (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
63 | Duverney | Duvernay formation tops SPE April 20 2021 | 07_Ostracod_Beds (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
64 | Duverney | Duvernay formation tops SPE April 20 2021 | 08_Jurassic_Top (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
65 | Duverney | Duvernay formation tops SPE April 20 2021 | 09_Montney_Top (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
66 | Duverney | Duvernay formation tops SPE April 20 2021 | 10_Permian_Top (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
67 | Duverney | Duvernay formation tops SPE April 20 2021 | 11_Wabamun (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
68 | Duverney | Duvernay formation tops SPE April 20 2021 | 12_Winterburn (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
69 | Duverney | Duvernay formation tops SPE April 20 2021 | 13_Woodbend (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
70 | Duverney | Duvernay formation tops SPE April 20 2021 | 14_Duvernay_Top (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
71 | Duverney | Duvernay formation tops SPE April 20 2021 | 15_Beaverhill (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
72 | Duverney | Duvernay formation tops SPE April 20 2021 | 16_Elk_Point (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
73 | Duverney | Duvernay formation tops SPE April 20 2021 | 17_Precambrian Basement (Surbiton)[SSTVD] (m) | Top depth of named Formation per well, measured vertically from sea level
74 | Duverney | Duvernay well headers SPE April 21 2021  | UWI  | Unique well Identifier
75 | Duverney | Duvernay well headers SPE April 21 2021  | Elevation Meters | Elevation measurement above mean sea level
76 | Duverney | Duvernay well headers SPE April 21 2021  | ElevationDatum | Elevation measurement point (Kelly Bushing)
77 | Duverney | Duvernay well headers SPE April 21 2021  | TD meters  | Total Depth of the well measured along the borehole in meters
78 | Duverney | Duvernay well headers SPE April 21 2021  | SurfaceLatitude_NAD83 | Latitude of well at surface, North American Datum of 1983
79 | Duverney | Duvernay well headers SPE April 21 2021  | SurfaceLongitude_NAD83 | Longitude of well at surface, North American Datum of 1983
80 | Duverney | Duvernay well headers SPE April 21 2021  | BottomLatitude_NAD83 | Latitude of well at bottom, North American Datum of 1983
81 | Duverney | Duvernay well headers SPE April 21 2021  | BottomLongitude_NAD83 | Longitude of well at bottom, North American Datum of 1983
82 | Duverney | Duvernay well headers SPE April 21 2021  | SurfaceLatitude_NAD27 | Latitude of well at surface, North American Datum of 1927 
83 | Duverney | Duvernay well headers SPE April 21 2021  | SurfaceLongitude_NAD27 | Longitude of well at surface, North American Datum of 1927 
84 | Duverney | Duvernay well headers SPE April 21 2021  | BottomLatitude_NAD27 | Latitude of well at bottom, North American Datum of 1927 
85 | Duverney | Duvernay well headers SPE April 21 2021  | BottomLongitude_NAD27 | Longitude of well at bottom, North American Datum of 1927 
86 | Duverney | SPE Duvernay production summary April 20 2021 | API    | Unique API number (US wells)
87 | Duverney | SPE Duvernay production summary April 20 2021 | Measured Depth (ft)    | Overall depth of a well - length of the well bore
88 | Duverney | SPE Duvernay production summary April 20 2021 | Total Vertical Depth (ft)    | Vertical distance from the bottom of the well to surface
89 | Duverney | SPE Duvernay production summary April 20 2021 | Spud Date    | The date when drilling began for the well
90 | Duverney | SPE Duvernay production summary April 20 2021 | Completion Date    | The date when well was completed
91 | Duverney | SPE Duvernay production summary April 20 2021 | First Production Month    | Month when the first production from the well was reported
92 | Duverney | SPE Duvernay production summary April 20 2021 | Elevation    | Elevation measured above a certain datum (ground or mean sea level)
93 | Duverney | SPE Duvernay production summary April 20 2021 | Oil Total Cum (bbl)    | Cumulative Oil Production at the time of reporting this data
94 | Duverney | SPE Duvernay production summary April 20 2021 | Gas Total Cum (mcf)    | Cumulative Gas Production at the time of reporting this data
95 | Duverney | SPE Duvernay production summary April 20 2021 | Water Total Cum (bbl)    | Cumulative Water Production at the time of reporting this data
96 | Duverney | SPE Duvernay production summary April 20 2021 | GOR Total Average    | Gas Oil Ratio
97 | Duverney | SPE Duvernay production summary April 20 2021 | Plug Date    | If the well reached the abondanment limit then when was it plugged
98 | Duverney | SPE Duvernay production summary April 20 2021 | First Production Date    | Date when the first production from the well was reported
99 | Duverney | SPE Duvernay production summary April 20 2021 | Elevation Drill Floor (ft)    | Elevation of drill floor above mean sea level
100 | Duverney | SPE Duvernay production summary April 20 2021 | Elevation Ground (ft)    | Elevation of ground level above mean sea level
101 | Duverney | SPE Duvernay production summary April 20 2021 | Elevation Kelly Bushing (ft)    | Elevation of kelly bushing above mean sea level
102 | Duverney | SPE Duvernay production summary April 20 2021 | Last Production Month    | The last production month reported
103 | Duverney | SPE Duvernay production summary April 20 2021 | Gas Maximum (mcf)    | Maximum amount of gas production at one time
104 | Duverney | SPE Duvernay production summary April 20 2021 | Gas Maximum Date    | Date the maximum gas production was recorded
105 | Duverney | SPE Duvernay production summary April 20 2021 | Oil Maximum (bbl)    | Maximum amount of oil production at one time
106 | Duverney | SPE Duvernay production summary April 20 2021 | Oil Maximum Date    | Date the maximum oil production was recorded
107 | Duverney | SPE Duvernay production summary April 20 2021 | Water Maximum (bbl)    | Maximum amount of water production at one time
108 | Duverney | SPE Duvernay production summary April 20 2021 | Water Maximum Date    | Date the maximum water production was recorded
109 | Duverney | SPE Duvernay production summary April 20 2021 | Yield Total Average    | Oil and Gas production versus total production
110 | Eaglebine | Eaglebine BHT TSC data for SPE April 21 2020 | UWI | Unique well Identifier
111 | Eaglebine | Eaglebine BHT TSC data for SPE April 21 2020 | SurfLat | Latitude coordinate of well at surface
112 | Eaglebine | Eaglebine BHT TSC data for SPE April 21 2020 | SurfLong | Longitude coordinate of well at surface
113 | Eaglebine | Eaglebine BHT TSC data for SPE April 21 2020 | TD (ft) | Total Depth of the well measured along the borehole in feet
114 | Eaglebine | Eaglebine BHT TSC data for SPE April 21 2020 | GL(ft) | Ground Level (from mean sea level)
115 | Eaglebine | Eaglebine BHT TSC data for SPE April 21 2020 | BHT_below sea level (ft) | Bottom Hole Temperature in the wellbore measured from sea level depth
116 | Eaglebine | Eaglebine BHT TSC data for SPE April 21 2020 | BHT_ subsurface (ft) | Bottom Hole Temperature in the wellbore measured from ground level depth
117 | Eaglebine | Eaglebine BHT TSC data for SPE April 21 2020 | BHTorMRT (maximum recorded temperature) oF | Bottom hole temperature or Maximum recorded Temperature
118 | Eaglebine | Eaglebine BHT TSC data for SPE April 21 2020 | TSC or ORT (time since circulation or original recorded time in hours) | Time since circulation
119 | Eaglebine | EagleBine Casing production summary for SPE April21 2020 | UWI | Unique well Identifier
120 | Eaglebine | EagleBine Casing production summary for SPE April21 2020 | MinCasingSize\ | The minimum casing size for a well
121 | Eaglebine | EagleBine Casing production summary for SPE April21 2020 | spuddate | The date when drilling begins for a well
122 | Eaglebine | EagleBine Casing production summary for SPE April21 2020 | completiondate | The date when well was completed
123 | Eaglebine | EagleBine Casing production summary for SPE April21 2020 | cumoil | Cumulative Oil Production at the time of reporting this data
124 | Eaglebine | EagleBine Casing production summary for SPE April21 2020 | cumgas | Cumulative Gas Production at the time of reporting this data
125 | Eaglebine | EagleBine Casing production summary for SPE April21 2020 | cumwater | Cumulative Water Production at the time of reporting this data
129 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | UWI | Unique well Identifier
130 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | Bottom_Hole_Location_X | X coordinates at bottom of wellbore
131 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | Bottom_Hole_Location_Y | Y coordinates at bottom of wellbore
132 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | Elevation(f) | Elevation measured above a certain datum (ground or mean sea level)
133 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | Elevation_Reference | Elevation measurement point (Kelly Bushing)
134 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | Latitude | Latitude coordinate
135 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | Longitude | Longitude coordinate
136 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | Total_Depth(f) | Total Depth of the well measured along the borehole in feet
137 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | X(f) | X coordinates
138 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | Y(f) | Y coordinates 
139 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 01_Wilcox_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
140 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 02_Midway_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
141 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 03_Navarro_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
142 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 04_Taylor_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
143 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 05_Anacacho_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
144 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 06_Austin_Chalk_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
145 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 07_Upper_Eagle_Ford_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
146 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 08_Lower_Eagle_Ford_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
147 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 09_Woodbine_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
148 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 10_Maness_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
149 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 11_Buda_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
150 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 12_Del_Rio_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
151 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 13_Georgetown_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
152 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 14_Edwards_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
153 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 15_Glen_Rose_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
154 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 16_Pearsal_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
155 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 17_James_Cow_Creek_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
156 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 18_Sligo_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
157 | Eaglebine | Eaglebine formation tops SPE April 20 2021 | 19_Cotton_Valley_MD_ft | Top depth of named Formation per well, measured along the wellbore in feet
158 | Eaglebine | Eaglebine mud weight SPE April 21 2021 | UWI | Unique well Identifier
159 | Eaglebine | Eaglebine mud weight SPE April 21 2021 | TD | Total Depth of the well measured along the borehole in feet
160 | Eaglebine | Eaglebine mud weight SPE April 21 2021 | KB | Kelly Bushing
161 | Eaglebine | Eaglebine mud weight SPE April 21 2021 | Mud Wt | Weight of drilling fluid
162 | Eaglebine | Eaglebine mud weight SPE April 21 2021 | MW@Depth(KB) | Depth at which mud weight is measured
163 | Eaglebine | Eaglebine well headers SPE April 21 2021 | td | Total Depth of the well measured along the borehole
164 | Eaglebine | Eaglebine well headers SPE April 21 2021 | Elevation | Elevation measured above a certain datum (ground or mean sea level)
165 | Eaglebine | Eaglebine well headers SPE April 21 2021 | ElevationDatum | Elevation measurement point (Kelly Bushing)
166 | Eaglebine | Eaglebine well headers SPE April 21 2021 | displayapi | Unique API number (US wells)
167 | Eaglebine | Eaglebine well headers SPE April 21 2021 | WGS84Latitude | World Geodetic System (WGS84) easting coordinates
168 | Eaglebine | Eaglebine well headers SPE April 21 2021 | WGS84Longitude | World Geodetic System (WGS84) northing coordinates
169 | Eaglebine | Eaglebine well headers SPE April 21 2021 | SurfLat | Latitude coordinate of well at surface
170 | Eaglebine | Eaglebine well headers SPE April 21 2021 | SurfLong | Longitude coordinate of well at surface
171 | Eaglebine | Eaglebine well headers SPE April 21 2021 | SurfaceLatitude_NAD83 | Latitude of well at surface, North American Datum of 1983
172 | Eaglebine | Eaglebine well headers SPE April 21 2021 | SurfaceLongitude_NAD83 | Longitude of well at surface, North American Datum of 1983
173 | Eaglebine | Eaglebine well headers SPE April 21 2021 | BottomLatitude_NAD83 | Latitude of well at bottom, North American Datum of 1983
174 | Eaglebine | Eaglebine well headers SPE April 21 2021 | BottomLongitude_NAD83 | Longitude of well at bottom, North American Datum of 1983
175 | Eaglebine | Eaglebine well headers SPE April 21 2021 | SurfaceLatitude_NAD27 | Latitude of well at surface, North American Datum of 1927 
176 | Eaglebine | Eaglebine well headers SPE April 21 2021 | SurfaceLongitude_NAD27 | Longitude of well at surface, North American Datum of 1927 
177 | Eaglebine | Eaglebine well headers SPE April 21 2021 | BottomLatitude_NAD27 | Latitude of well at bottom, North American Datum of 1927 
178 | Eaglebine | Eaglebine well headers SPE April 21 2021 | BottomLongitude_NAD27 | Longitude of well at bottom, North American Datum of 1927 
179 | Eaglebine | SPE Eaglebine production summary April 20 2021 | API    | Unique API number (US wells)
180 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Measured Depth (ft)    | Total Depth of the well measured along the borehole in feet
181 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Total Vertical Depth (ft)    | Vertical distance from the bottom of the well to surface in feet
182 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Spud Date    | The date when drilling began for the well
183 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Completion Date    | The date when well was completed
184 | Eaglebine | SPE Eaglebine production summary April 20 2021 | First Production Month    | Month when the first production from the well was reported
185 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Elevation    | Elevation measured above a certain datum (ground or mean sea level)
186 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Oil Total Cum (bbl)    | Cumulative Oil Production at the time of reporting this data
187 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Gas Total Cum (mcf)    | Cumulative Gas Production at the time of reporting this data
188 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Water Total Cum (bbl)    | Cumulative Water Production at the time of reporting this data
189 | Eaglebine | SPE Eaglebine production summary April 20 2021 | GOR Total Average    | Gas Oil Ratio
190 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Plug Date    | If the well reached the abondanment limit then when was it plugged
191 | Eaglebine | SPE Eaglebine production summary April 20 2021 | TD Date    | Date when drilling got to the bottom depth of the well
192 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Drilling Days    | Number of days it took to drill the well
193 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Last Production Month    | The last production month reported
194 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Gas Maximum (mcf)    | Maximum amount of gas production at one time
195 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Gas Maximum Date    | Date the maximum gas production was recorded
196 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Oil Maximum (bbl)    | Maximum amount of oil production at one time
197 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Oil Maximum Date    | Date the maximum oil production was recorded
198 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Water Maximum (bbl)    | Maximum amount of water production at one time
199 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Water Maximum Date    | Date the maximum water production was recorded
200 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Liquid Injection Cum (bbl)    | Cumulative liquid injected into the well at the time of reporting this data
201 | Eaglebine | SPE Eaglebine production summary April 20 2021 | Gas Injection Cum (mcf)    | Cumulative gas injected into the well at the time of reporting this data
