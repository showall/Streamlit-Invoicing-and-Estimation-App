import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import os
from requests_html import HTMLSession
from requests_file import FileAdapter
import shutil
from jinja2 import Environment, PackageLoader, select_autoescape, FileSystemLoader
import matplotlib
matplotlib.use('Agg')
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import plotly.express as px
import io 
from datetime import datetime
from pathlib import Path
import pdfkit
import os
from pyhtml2pdf import converter
import chromedriver_autoinstaller as chromedriver

chromedriver.install()

STREAMLIT_STATIC_PATH = Path(st.__path__[0]) / 'static'
CSS_PATH = (STREAMLIT_STATIC_PATH / "assets/css")
if not CSS_PATH.is_dir():
    CSS_PATH.mkdir()

JS_PATH = (STREAMLIT_STATIC_PATH / "assets/js")
if not JS_PATH.is_dir():
    JS_PATH.mkdir()

css_file = CSS_PATH / "style.css"
js_file = JS_PATH / "script.js"
if not css_file.exists():

	shutil.copy("style.css", css_file)
	shutil.copy("script.js", js_file)
else:
#	os.remove(css_file)
	shutil.copy("style.css", css_file)
	shutil.copy("script.js", js_file)

env = Environment(loader=FileSystemLoader("."), autoescape=select_autoescape())
template = env.get_template("invoice.html")
template_2 = env.get_template("invoice_template_for_report.html")

#Set title
st.set_page_config(layout="wide")


with open(css_file,'r') as f: 
	css_data = f.read()
st.markdown(f"""<style>   
{css_data}</style>""", unsafe_allow_html=True)	
image = Image.open('business.jpg')

def make_grid(cols,rows):
    grid = [0]*cols
    for i in range(cols):
        with st.container():
            grid[i] = st.columns(rows)
    return grid


mygrid = make_grid(1,3)
mygrid[0][0].image(image,use_column_width=False, width=80)
mygrid[0][1].markdown('### Good Good Company')

def findme():
	return st.text("function call")

# making grids

# mygrid[2][2].write('22')
# mygrid[3][3].write('33')
# mygrid[4][4].write('44')


# mygrid = [[],[]]
# with st.container():
#     mygrid[0] = st.columns(2)
# with st.container():
#     mygrid[1] = st.columns(2)

# mygrid[0][0].write('Caption for first chart')
# mygrid[0][1].line_chart((1,0), height=100)
# mygrid[1][0].write('Caption for second chart')
# mygrid[1][1].line_chart((0,1), height=100)


def main():
	with st.sidebar:
		choose = option_menu("Menu", ["Invoice Report", "Option 2"],
							#icons=['house', 'camera fill', 'kanban', 'book','person lines fill'],
							menu_icon="app-indicator", default_index=0,
			# 				styles={
			# "container": {"padding": "5!important", "background-color": "#fafafa"},
			# "icon": {"color": "orange", "font-size": "25px"}, 
			# "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
			# "nav-link-selected": {"background-color": "#02ab21"},
#		}
		)

	if choose =="Invoice Report":
		df = {}
		original_df = pd.DataFrame({})
		idx = 0
		submitted = False
		tab1, tab2 = st.tabs(["Upload","Result"])

		activities=['Customer1','Customer2','Customer3','Customer4']
		option=st.sidebar.selectbox('Selection option:',activities)		
		with tab1:
		#DEALING WITH THE EDA PART
			col1, col2, col3 = st.columns((0.1,2,0.1))

						# code to draw the single continent data
			with st.container():
				if option=='Customer1':
					st.subheader("Customer1")

					data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])

					if data is not None:
						st.success("Data successfully loaded")
						df=pd.read_csv(data, skiprows=0, parse_dates= ["DATE"], dayfirst=True)
						st.dataframe(df.head(50))
						st.markdown("### Add Pricing Features:")
						#with st.form("my form"):
						st.markdown("#### Rate Prices:")

						add_more_pricing = True
						data_dict = {}
						price_data = {}

						idx = 0
						while add_more_pricing is True:
							if idx == 0:
								data_dict[f"denom_{idx+1}"] = "first"
							else :
								data_dict[f"denom_{idx+1}"]  = "next"

							idx += 1								
							price_data[f"pricing_mode_a{idx}"] = 0
							price_data[f"pricing_mode_b{idx}"] = 0
							price_data[f"pricing_mode_c{idx}"] = 0
							price_data[f"pricing_mode_d{idx}"] = 0
							price_data[f"price_i{idx}"] = 0
							price_data[f"price_ii{idx}"] = 0
							price_data[f"price_iii{idx}"] = 0
							price_data[f"price_iv{idx}"] = 0
							# price_data[""] = 
							# price_data[""] = 
							# price_data[""] = 
							# price_data[""] = 

							with st.expander("Price Setting"):
								st.text(f"Price {idx}")
								first_grid = make_grid(2,2)
								data_dict[f"bd_{idx}"] = first_grid[0][0].selectbox("Quantity Split Per Usage",
								(
								"Prorated-Block",
								"Reserved-Block",
								"Pay-per-Use",
								), key= f"bd_{idx}")
								if data_dict[f"bd_{idx}"] == "Pay-per-Use":
									data_dict[f"values_{idx}"] = first_grid[0][1].number_input(
									f'Pricing Applied to the {data_dict[f"denom_{idx}"] } % Volume (0.00 to 100.0%)', key=f"values_{idx}")
									st.write(f'Applicable to the {data_dict[f"denom_{idx}"] } :', data_dict[f"values_{idx}"])
									if data_dict[f"values_{idx}"] > 100 or data_dict[f"values_{idx}"] <0:
										st.error("split has to be between 0.00 to 100.0%", icon="🚨")

								elif data_dict[f"bd_{idx}"]  == "Prorated-Block": 
									data_dict[f"values_{idx}"] = first_grid[0][1].number_input(
									f'Pricing Applied to the {data_dict[f"denom_{idx}"]} Amount of Volume', key=f"values_2{idx}")
									st.write(f'Applicable to the {data_dict[f"denom_{idx}"]}:', data_dict[f"values_{idx}"])
									if data_dict[f"values_{idx}"] < 0:
										st.error("block has to be greater than 0.00", icon="🚨")					
								
								else:
									data_dict[f"values_{idx}"] = first_grid[0][1].number_input(
									f'Pricing Applied to the {data_dict[f"denom_{idx}"]} Amount of Volume ', key=f"values_3{idx}")

									data_dict[f"values_2{idx}"] = first_grid[1][1].number_input(
									f'CAP Applied to the {data_dict[f"denom_{idx}"]} Amount of Volume ', key=f"values_4{idx}")

									st.write(f'Applicable to the {data_dict[f"denom_{idx}"]} capped at {data_dict[f"values_2{idx}"]}  per period:', data_dict[f"values_2{idx}"])
									if data_dict[f"values_{idx}"] < 0:
										st.error("block has to be greater than 0.00", icon="🚨")					
									if data_dict[f"values_{idx}"] < 0:
										st.error("block has to be greater than 0.00", icon="🚨")									
						
								data_dict[f"tos_{idx}"] = st.selectbox("Time of Use Mode",
								(
								"Flat",
								"Weekday/Weekend",
								"10pm-8am/8am-10pm",
								""
								), key= f"tos_{idx}")

							#	with st.expander("Price Settings"):
								if data_dict[f"tos_{idx}"] == "Flat":
									grid_2 = make_grid(1,2)
									price_data[f"pricing_mode_a{idx}"]  = grid_2[0][0].selectbox("Pricing Mode",
									(
									"On Demand Pricing",
									"Reserved Pricing",										
									"Special Pricing",
									""
									), key= f"pmn_{idx}")
								
									if price_data[f"pricing_mode_a{idx}"]  == "Reserved Pricing":
										price_data[f"price_i{idx}"] = grid_2[0][1].number_input("What Price?", value=199, key= f"fxdr_{idx}")
									elif price_data[f"pricing_mode_a{idx}"]  == "Special Pricing":
										price_data[f"price_i{idx}"] = grid_2[0][1].number_input("What Price?", key= f"fxds_{idx}")			

								elif data_dict[f"tos_{idx}"] == "Weekday/Weekend":
									grid_2 = make_grid(2,2)									
									price_data[f"pricing_mode_a{idx}"] = grid_2[0][0].selectbox("Pricing Mode Weekday",
									(
									"On Demand Pricing",
									"Reserved Pricing",									
									"Special Pricing",
									""
									), key= f"pmd_{idx}")						

									if price_data[f"pricing_mode_a{idx}"] == "Reserved Pricing":
										price_data[f"price_i{idx}"]  = grid_2[0][1].number_input("What Price?", value=199, key= f"wkdr_{idx}")
									elif price_data[f"pricing_mode_a{idx}"] == "Special Pricing":
										price_data[f"price_i{idx}"] = grid_2[0][1].number_input("What Price?", key= f"wkds_{idx}")

									price_data[f"pricing_mode_b{idx}"] = grid_2[1][0].selectbox("Pricing Mode Weekend",
									(
									"On Demand Pricing",
									"Reserved Pricing",										
									"Special Pricing",
									""
									), key= f"pmw_{idx}")	

									if price_data[f"pricing_mode_b{idx}"] == "Reserved Pricing":
										price_data[f"price_ii{idx}"] = grid_2[1][1].number_input("What Price?", value=179, key= f"wedr_{idx}")
									elif price_data[f"pricing_mode_b{idx}"] == "Special Pricing":
										price_data[f"price_ii{idx}"] = grid_2[1][1].number_input("What Price?", key= f"weds_{idx}")
							#		st.checkbox("Do Something", key= f"pricing_{idx}")

								elif data_dict[f"tos_{idx}"] == "10pm-8am/8am-10pm":
									grid_2 = make_grid(2,2)	
									price_data[f"pricing_mode_a{idx}"] = grid_2[0][0].selectbox("Pricing Mode Peak Hours",
									(
									"On Demand Pricing",
									"Reserved Pricing",										
									"Special Pricing",
									""
									), key= f"pmph_{idx}")						

									if price_data[f"pricing_mode_a{idx}"]  == "Reserved Pricing":
										price_data[f"price_i{idx}"] = grid_2[0][1].number_input("What Price?", value=199, key= f"pr_{idx}")
									elif price_data[f"pricing_mode_a{idx}"]  == "Special Pricing":
										price_data[f"price_i{idx}"] = grid_2[0][1].number_input("What Price?", key= f"ps_{idx}")

									price_data[f"pricing_mode_b{idx}"]  = grid_2[1][0].selectbox("Pricing Mode Off Peak Hours",
									(
									"On Demand Pricing",
									"Reserved Pricing",										
									"Special Pricing",
									""
									), key= f"pmoph_{idx}")	

									if price_data[f"pricing_mode_b{idx}"] == "Reserved Pricing":
										price_data[f"price_ii{idx}"] = grid_2[1][1].number_input("What Price?", value=179, key= f"opr_{idx}")
									elif price_data[f"pricing_mode_b{idx}"] == "Special Pricing":
										price_data[f"price_ii{idx}"] = grid_2[1][1].number_input("What Price?", key= f"ops_{idx}")
								#		st.checkbox("Do Something", key= f"pricing_{idx}")
								
							grid = make_grid(1,2)								
							add_more_pricing = grid[0][0].checkbox("Add Another Pricing", key = f"add_{idx}")


						st.markdown("#### Apply Discounts:")
						mygrid_3 = make_grid(2,4)
						feature_2_1 = mygrid_3[0][0].checkbox("Add Discount 1")
						feature_2_2 = mygrid_3[0][1].checkbox("Add Discount 2")
						submitted = st.button(label="submit")
						if submitted:
							st.text("Calculation done, please see the result tab.")

	
		with tab2:
			col1, col2, col3 = st.columns((0.1,2,0.1))

			with col2:
				if submitted:
					html_string = "<h3>Results       </h3>"
					st.markdown(html_string, unsafe_allow_html=True)
					def split_day_1(x):
						if x.dayofweek != 0 and x.dayofweek !=6:
							return 1
						else :
							return 2

					def split_time_1(x):
						if x.hour < 8 or x.hour > 22 :
							return 2
						else:
							return 1

					df['part_split_1'] = 1
					df['time_split_1'] = df["DATE"].apply(split_time_1)
					df["day_split_1"] = df["DATE"].apply(split_day_1)


					dict_col_to_use = {"Flat" : 'part_split_1',
							'10pm-8am/8am-10pm': 'time_split_1',
							"Weekday/Weekend" : "day_split_1"}		

					def volume_break(df, time_mode = "Flat"):

						volume_1 = df.apply(lambda x: x.USAGE if x[dict_col_to_use[time_mode]]==1 else 0, axis=1) 
						volume_2 = df.apply(lambda x: x.USAGE if x[dict_col_to_use[time_mode]]==2 else 0, axis=1) 
						volume_3 = df.apply(lambda x: x.USAGE if x[dict_col_to_use[time_mode]]==3 else 0, axis=1) 
						volume_4 = df.apply(lambda x: x.USAGE if x[dict_col_to_use[time_mode]]==4 else 0, axis=1)
						return volume_1, volume_2, volume_3, volume_4

					def volume_billable(idx,volume_1, volume_2, volume_3, volume_4, how, volume_left):
						volume_used = dict()
						volume_lefted = dict()
						if how == "Pay-per-Use":
							volume_used["volume_used_1"] = volume_1 * data_dict[f"values_{idx}"]/100 
							volume_used["volume_used_2"]  = volume_2 * data_dict[f"values_{idx}"] /100
							volume_used["volume_used_3"]  = volume_3 * data_dict[f"values_{idx}"]/100 
							volume_used["volume_used_4"] = volume_4 * data_dict[f"values_{idx}"]/100 

						elif how == "Prorated-Block" :
							volume_used["volume_used_1"]= pd.Series([data_dict[f"values_{idx}"] \
								/len(volume_1[volume_1 != 0]) if x !=0 else 0 for x in volume_1])
							volume_used["volume_used_2"] = pd.Series([data_dict[f"values_{idx}"] \
								/len(volume_2[volume_2 != 0]) if x !=0 else 0 for x in volume_2])
							volume_used["volume_used_3"] = pd.Series([data_dict[f"values_{idx}"] \
								/len(volume_3[volume_3 != 0]) if x !=0 else 0 for x in volume_3])
							volume_used["volume_used_4"] = pd.Series([data_dict[f"values_{idx}"] \
								/len(volume_4[volume_4 != 0]) if x !=0 else 0 for x in volume_4])

						else :
							volume_used["volume_used_1"] = volume_1
							volume_used["volume_used_2"] = volume_2
							volume_used["volume_used_3"] = volume_3
							volume_used["volume_used_4"] = volume_4 
							if volume_left != None:
								running_volume_dict = volume_left
							else :
								running_volume_dict = volume_used

							running_volume_dict["volume_used_1"] = np.minimum(volume_used["volume_used_1"] * data_dict[f"values_2{idx}"] /100, running_volume_dict["volume_used_1"]) 
							running_volume_dict["volume_used_2"] = np.minimum(volume_used["volume_used_1"] * data_dict[f"values_2{idx}"] /100, running_volume_dict["volume_used_2"]) 
							running_volume_dict["volume_used_3"] = np.minimum(volume_used["volume_used_1"] * data_dict[f"values_2{idx}"] /100, running_volume_dict["volume_used_3"]) 
							running_volume_dict["volume_used_4"] = np.minimum(volume_used["volume_used_1"] * data_dict[f"values_2{idx}"] /100, running_volume_dict["volume_used_4"]) 					

							u= 0

							for series in running_volume_dict.values():
								u += 1
								empty = []
								volume_alloted = data_dict[f"values_{idx}"]
								for i,row in series.items():
									empty.append(min(volume_alloted,row))
									volume_alloted = max(volume_alloted-row,0)
									volume_used[f"volume_used_{u}"] = pd.Series(empty)

						volume_lefted["volume_used_1"] = volume_1 - volume_used["volume_used_1"]
						volume_lefted["volume_used_2"] = volume_2 - volume_used["volume_used_2"]
						volume_lefted["volume_used_3"] = volume_3 - volume_used["volume_used_3"]
						volume_lefted["volume_used_4"] = volume_4 - volume_used["volume_used_4"]

						return volume_used, volume_lefted
					def determine_price_volume(df, price_data, price, volume):
						if price_data == "On Demand Pricing":
							rate = df["USAGE"]
						else :
							rate = price
						
						billable = rate * volume
						return billable

					def unbillable_amount ():
						pass
					datafile = dict()
					original_df = df[["DATE","USAGE"]]
					original_df.columns = ["Date", "Used_Vol"]	
					unbilled = df["USAGE"]
					for id in range(1,idx+1):
						st.write(f"Tier {id} Pricing")
						volume_1, volume_2, volume_3, volume_4 = volume_break(df, 
																time_mode = data_dict[f"tos_{id}"] )			
						try:
							left_volume = volume_left
						except:
							left_volume = None
						volume_used, volume_left = volume_billable(id,
																	volume_1, 
																	volume_2, 
																	volume_3, 
																	volume_4, 
																	data_dict[f"bd_{id}"],
																	left_volume)

						volume_1, volume_2, volume_3, volume_4 = volume_used.values()
						new_df = pd.concat([df.DATE,volume_1, volume_2, volume_3, volume_4 ], axis=1)
						new_df.columns = ["Time",f"Vol{id}_Tier1", f"Vol{id}_Tier2", f"Vol{id}_Tier3", f"Vol{id}_Tier4"]
						
						st.dataframe(new_df)
						for col in [f"Vol{id}_Tier1", f"Vol{id}_Tier2", f"Vol{id}_Tier3", f"Vol{id}_Tier4"]:
							unbilled = np.maximum(unbilled - new_df[col],0)					
						#revenue_calculation:
						id2 = 0
						for i in zip(["a","b","c","d"],["i","ii","iii","iv"]):
							id2 += 1	
						# st.write(f"S$ {price_data}")						 
							billable = determine_price_volume(df, price_data[f"pricing_mode_{i[0]}{id}"], price_data[f"price_{i[1]}{id}"], new_df[f"Vol{id}_Tier{id2}"])
							new_df[f"Revenue{id}_Tier{id2}"] = billable
						original_df = pd.concat([original_df, new_df], axis=1)
						original_df = original_df.drop("Time", axis=1)				
					original_df["Unbilled"] = unbilled				
					total_sum = 0
					st.dataframe(original_df)		
					count_price_plan = 0					
					for id in range(1,idx+1):
						subtotal_sum = 0								
						for id2 in range(1,5):	
							volume_ratio = original_df[f"Vol{id}_Tier{id2}"]
							subtotal = original_df[f"Revenue{id}_Tier{id2}"]
							billable_sum = sum(subtotal.values)
							volume_sum = sum(volume_ratio.values)
							subtotal_sum = subtotal_sum + billable_sum						
							total_sum = total_sum + billable_sum
							if volume_sum != 0 :
								count_price_plan = count_price_plan + 1
								datafile[count_price_plan] = dict() 								
								if data_dict[f'bd_{id}'] == "Pay-per-Use":
									piece = "units"
								else:
									piece = "units per contracted block"
								datafile[count_price_plan]["Timeband"] =  	data_dict[f'tos_{id}']	
								datafile[count_price_plan]["Tier"] =  	f"Tier {id2}"
								datafile[count_price_plan]["Plan"] =  	data_dict[f'bd_{id}'] 									
								datafile[count_price_plan]["Volume_Used"] =  	f"{volume_sum:,.4f} {piece}"
								datafile[count_price_plan]["Rate"] =  	f"{billable_sum/volume_sum:,.4f}"
								datafile[count_price_plan]["Subtotal"] =  	f"{billable_sum/1:,.2f}"
								if data_dict[f'tos_{id}']== "Flat":
									st.markdown(f" {data_dict[f'tos_{id}']} PLAN  {data_dict[f'bd_{id}']}     {volume_sum:,.4f} {piece}    @rate {billable_sum/volume_sum:,.4f}      S$ {billable_sum:,.2f}")
								else :
									st.write(f" {data_dict[f'tos_{id}']} PLAN  {data_dict[f'bd_{id}']}           S$ {subtotal_sum:,.2f}")
									st.write(f"       Tier {id2}     {volume_sum:,.4f} {piece}    @rate {billable_sum/volume_sum:,.4f}      S$ {billable_sum:,.2f}")
					st.write(datafile)
					unbilled_vol = sum(original_df["Unbilled"])
					original_df["ubilled_amount"] = determine_price_volume(df, "On Demand Pricing", 0, original_df["Unbilled"])
					unbilled_rate = sum(determine_price_volume(df, "On Demand Pricing", 0, original_df["Unbilled"]).values)
					total_sum = total_sum + unbilled_rate		
					st.write(f"       Uncontracted     {unbilled_vol:,.4f}   @rate {unbilled_rate/unbilled_vol:,.4f}      S$ {unbilled_rate:,.2f}")
					datafile[count_price_plan+1] = dict()
					datafile[count_price_plan+1]["Timeband"] =  	"-"	
					datafile[count_price_plan+1]["Tier"] =  	f"-"
					datafile[count_price_plan+1]["Plan"] =  	"Uncontracted" 									
					datafile[count_price_plan+1]["Volume_Used"] =  	f"{unbilled_vol:,.4f} units "
					datafile[count_price_plan+1]["Rate"] =  	f"{unbilled_rate/unbilled_vol:,.4f}"
					datafile[count_price_plan+1]["Subtotal"] =  	f"{unbilled_rate/1:,.2f}"

					from_ = list(original_df["Date"])[0].strftime("%m/%d/%Y")
					to_ = list(original_df["Date"])[-1].strftime("%m/%d/%Y")
					invoice_id = "17183921334"
					html_data = template.render(
						invoice_number= invoice_id,
						date_generated = datetime.now().strftime("%m/%d/%Y"),
						Some_Company = "Gerard_SHO_Company",
						datafile = datafile,
						total=f"{total_sum:,.2f}",
						from_when = from_,
						to_when = to_ ,
						amount_due = f"{unbilled_rate/1:,.2f}"
					)

					# with open("saving.html",'r') as f: 
					# 	html_data = f.read()
					st.markdown(html_data, unsafe_allow_html=True)				

				## Show in webpage
				# st.header("Show an external HTML")
				# st.write(html_data)
				# st.markdown(html_data, unsafe_allow_html=True)
				#st.components.v1.html(html_data,height=800, width=400)
					with st.container():
						#col1, col2, col3 = st.columns((0.001,2,0.001))

							# code to draw the single continent data

					#	with col2:
						st.header("Billable Consumption")
						bar_plot_y = []
						for name in original_df.columns[2:]:
							if "Vol" in name and original_df[name].any()!=0:
								bar_plot_y.append(name)
							else:
								continue
								
						bar_plot_y.append("Unbilled")
						st.info("")		
						fig = px.bar(original_df,x="Date",y=bar_plot_y, barmode="stack",
						color_discrete_sequence=px.colors.qualitative.Alphabet).update_layout(legend={"title":"Tier Plan"})
						fig.add_scatter(x=original_df['Date'], y=original_df['Used_Vol'], mode='lines',line=go.scatter.Line(color="gray"), name="Uncontracted Volume")				

						fig.update_layout(legend=dict(
												yanchor="top",
												y=0.99,
												xanchor="left",
												x=0.01
											))
						st.plotly_chart(fig, use_container_width=True)
						image_1 = 'assets/images/abc.png'
						st.header("Gerard_SHO_Company")  
						st.info("")
						fig.write_image(image_1, engine="kaleido")
						fig = px.scatter(
							df,
							x="DATE",
							y="USAGE",
						).update_layout(legend={"title":"Consumption"})
						fig.add_scatter(x=original_df['Date'], y=original_df["Used_Vol"], mode='lines',line=go.scatter.Line(color="blue"), name="Consumption")				
						fig.update_layout(legend=dict(
												yanchor="top",
												y=0.99,
												xanchor="left",
												x=0.01
											))

						#fig.layout.xaxis.rangeslider.visible = False
						#fig.layout.yaxis.rangeslider.visible = False
						image_2 = 'assets/images/def.png'
						fig.write_image(image_2, engine="kaleido")
						st.plotly_chart(fig, use_container_width=True)		

						#fig.write_image(image_2)

						html_data_for_report = template_2.render(
						invoice_number= invoice_id,
						date_generated = datetime.now().strftime("%m/%d/%Y"),
						Some_Company = "Gerard_SHO_Company",
						datafile = datafile,
						total=f"{total_sum:,.2f}",
						from_when = from_,
						to_when = to_ ,
						amount_due = f"{unbilled_rate/1:,.2f}",
						image_1 = image_1,
						image_2 = image_2
					)

						with open('saving.html', 'w+') as f:
							f.write(html_data_for_report)

						st.markdown("<h4>Notes</h4>", unsafe_allow_html=True)
						st.markdown("<p>A finance charge of 1.5% will be made on unpaid balances after 30 days.</p>", unsafe_allow_html=True)


						###############REMOVE IT IF DEPLOYING FROM PROPRIETARY PLATFORM THEN CAN DOWNLOAD PDF DIRECTLY 

						# path = os.path.abspath('saving.html')
						# converter.convert(f'file:///{path}', f'sample.pdf'
						# 					, print_options={"scale": 0.94, "pageRanges" : "1-2"} 
						# 					)


						#OPTIONAL

						path_wkhtmltopdf = "wkhtmltopdf/bin/wkhtmltopdf.exe"
						config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
						pdf = pdfkit.from_string(html_data_for_report, "output.pdf", configuration=config,  options={"enable-local-file-access": ""})
						# to create download on clicking


						with open("sample.pdf", "rb") as pdf_file:
							PDFbyte = pdf_file.read()


						st.download_button(label="Export_Report",
											data=PDFbyte,
											file_name="test.pdf",
											mime='application/octet-stream')


if __name__ == '__main__':
	main() 



