import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import seaborn as sns
import time


@st.cache
def load_data():
	df = pd.read_csv('updated_sample_with_boroughs.csv')
	return df

df = load_data()


st.title('Maximising revenue as a New York City Taxi Driver')
#############################################################################

st.write('This small web application explores the New York City Taxi trips dataset from August 2013. In particular, we aim to develop a strategy for a taxi driver to maximise their revenue. To this end, our exploration of the data leads us to develop 3 "rules" for a driver who wants to maximise their revenue (in case you wanted to go back in time to August 2013). These are: ')
st.write('1. Stay within Manhattan (in particular, the Lower-Manhattan, Midtown and Upper East Side areas).')
st.write('2. Be most active from 7pm to 12am and optimise their number of pick-ups (so shorter trips, rather than longer ones).')
st.write('3. At each point in time, choose the "highest value" routes.')

st.write('The remainder of this web app is divided into 3 sections, addressing the rationale for each rule.')

#############################################################################
st.header('Rule 1: Stay within Manhattan')

st.write('New York City is divided into 5 Boroughs. To familiarise yourself with them, you can expand the section immediately underneath to view a map of the boroughs.')

with st.beta_expander("New York City Boroughs"):
	from PIL import Image
	image = Image.open('nyc_boroughs_pic.png')
	st.image(image, caption='New York City Boroughs', use_column_width=True)


high_level_hours = df.groupby(['pickup_hour', 'pickup_borough']).agg({'trip_id': pd.Series.nunique})
high_level_hours.reset_index(inplace=True)

fig = px.bar(high_level_hours, x="pickup_hour", y="trip_id", 
				  #xaxis_title="Pick-up Hour",
                  #yaxis_title="Number of Pick-ups",
	color="pickup_borough", title="Busiest pick-up locations by Borough",
	height=600, width = 900)

st.plotly_chart(fig)

st.markdown('The chart above clearly shows that the vast majority of trips originate in Manhattan. In fact, Manhattan accounts for **89% of all pick-ups**. Thus, we will focus our attention to trips within Manhattan. You can once again, familiarise yourself with the geography of Manhattan by expanding the next section below.')

##############################################################################

with st.beta_expander("Manhattan Map "):
	image = Image.open('manhattan-map.jpg')
	st.image(image, caption='Manhattan', use_column_width=True)


man_hours = df[df['pickup_dropoff']=='manhattan-manhattan'].groupby(['pickup_hour', 'pickup_manhattan']).agg({'trip_id': pd.Series.nunique})
man_hours.reset_index(inplace=True)

fig = px.bar(man_hours, x="pickup_hour", y="trip_id", 
	color="pickup_manhattan", title="Busiest pick-up locations in Manhattan",
	height=600, width = 900)

st.plotly_chart(fig)

st.markdown('Repeating the format of the 1st bar chart, but focused only on Manhattan and areas within Manhattan we observe that **73% of all pick-ups in Manhattan originate from Lower-Manhattan, Midtown, Upper East Side areas.** This is perhaps not too surprising, as these areas are home to Wall St and the Financial District, Times Square, the Empire State Building and most of the tourist attractions in New York City. In the next chart we look more specifically at the starting and ending location of trips within Manhattan.')


fig = go.Figure(data=[go.Histogram(y=df[df['pickup_dropoff']=='manhattan-manhattan']["pickup_dropoff_man"])])

fig.update_layout(yaxis={'categoryorder':'total ascending'}, 
	title = "Top Pick-up, Drop-off locations in Manhattan",
	height=600, width = 800)

st.plotly_chart(fig)

with st.beta_expander("Key Takeaways: "):
	st.write('* We see that the most common trips are:')

	st.write('1. Lower Manhattan to Lower Manhattan')
	st.write('2. Midtown to Midtown')
	st.write('3. Lower-Manhattan to Midtown (and back)')
	st.write('4. Upper East Side to Upper East Side')
	st.write('5. Midtown to Upper East Side')

	st.markdown('Given these observations, ** I would recommend that the driver should stay within Manhattan, and more specifically, try to keep their trips within the Lower-Manhattan, Midtown and Upper East Side areas.** Of course, there is always a bit of a chicken and the agg argument here, is it that all the potential customers are indeed in these areas, or is it simply because cab drivers like to work from these areas? Our recommendation assumes the former case.')

	st.write('In fact, the above charts also lead us to justify part of rule 2, that drivers should be most active in the evenings. The charts above clearly show that the most pick-up times are in the evening from around 7pm to 12am.')
 

####################################################################################################
st.header('Rule 2: Be most active from 7pm to 12am and optimise number of pick-ups')

st.write('Whilst we have somewhat justified the first part of rule 2, we need to justify our rationale for optimising the number of pick-ups. A driver\'s revenue for a day can be seen to be a function of two things: the number of trips they make and the fare charged for each trip. The fare charged for each trip would be a function of the trip duration and the distance.')

st.write('To maximise revenue, there are two obvious options: either maximise the number of pick-ups, or maximise the fare amount of each trip (and thus either maximise distance or maximise duration of trips). There is an obvious trade off between each option. The more trips we make, the shorter duration they are, and the longer the duration of trips, the less number of pick-ups we would expect to make.')


st.write('The next chart allows us to explore for each trip within Manhattan, the movement of average fares and trip times at each hour of the day.')
#st.subheader('Variance of trip times and fares by pick-up, drop-off locations in Manhattan')

df_manhattan = df[df['pickup_dropoff']=='manhattan-manhattan']


top_10_manhattan_trips = df_manhattan.groupby(['pickup_dropoff_man', 'pickup_hour']).agg({'trip_id': pd.Series.nunique,
                                                        'trip_in_mins': 'mean',
                                                        ' fare_amount': 'mean',
                                                        ' total_amount': 'mean'}).sort_values(by='trip_id', ascending=False)
top_10_manhattan_trips.reset_index(inplace=True)

top_10_manhattan_trips.rename(columns={'trip_id': 'number_of_trips',
                                    'trip_in_mins': 'avg_trip_time_mins',
                                    ' fare_amount': 'avg_fare_amount',
                                    ' total_amount': 'avg_total_amount'}, inplace=True)

top_10_manhattan_trips = top_10_manhattan_trips.head(240)


top_10_manhattan_trips['avg_number_of_trips_in_hour'] = top_10_manhattan_trips['number_of_trips']/31

top_10_manhattan_trips['route_value_for_hour'] = top_10_manhattan_trips['avg_total_amount']*top_10_manhattan_trips['avg_number_of_trips_in_hour']

####################################

#fig = px.histogram(df[df['pickup_dropoff']=='manhattan-manhattan'], x=" fare_amount", histnorm='probability density',
#			 title="Distribution of Fare Amount for Manhattan trips.",
#			 width=900,
#                  height=600)

#st.plotly_chart(fig)

#st.write('* We see that the fare amounts are heavily skewed right, with the vast majority of fares between $5 and $20 dollars. Again, consistent with the observation that most trips are within the Lower-Manhattan, Midtown and Upper East Side areas, which are all geographically close.')


#options1=list(top_10_manhattan_trips['pickup_dropoff_man'].unique())
#route_selection1 = st.selectbox('Select pick-up, drop-off locations:', options1)

#fig = px.histogram(df[ (df['pickup_dropoff']=='manhattan-manhattan') & (df['pickup_dropoff_man']== route_selection1)], x=" fare_amount", histnorm='probability density',
#			 color='pickup_dropoff_man',
#			 title="Distribution of Fare Amount for: " + str(route_selection1) )

#st.plotly_chart(fig)



####################################

options1=list(top_10_manhattan_trips['pickup_dropoff_man'].unique())
route_selection = st.selectbox('Select pick-up, drop-off:', options1)

y_value = top_10_manhattan_trips[top_10_manhattan_trips['pickup_dropoff_man']==route_selection ]['avg_trip_time_mins']
y2_value = top_10_manhattan_trips[top_10_manhattan_trips['pickup_dropoff_man']== route_selection]['avg_fare_amount']

trace = go.Bar(x= top_10_manhattan_trips[top_10_manhattan_trips['pickup_dropoff_man']== route_selection]['pickup_hour'], 
               y = y_value,
              opacity=0.7,
              name= 'Average trip time in minutes')

trace2 = go.Scatter(x= top_10_manhattan_trips[top_10_manhattan_trips['pickup_dropoff_man']== route_selection]['pickup_hour'], 
               		y = y2_value,
                    mode='markers',
                   name='Average fare amount in dollars')


g = go.FigureWidget(data= [trace, trace2],
                    layout=go.Layout(
                        title=dict(
                            text='Average trip times and Fare amounts each hour for: ' + str(route_selection)
                        ),
                        barmode='overlay',
                        width=1000,
                  		height=600
                    ))


st.plotly_chart(g)

st.write('Standard deviation of average trip times for ' + str(route_selection))
st.write(y_value.std())

st.write('Standard deviation of average fare amounts for ' + str(route_selection))
st.write(y2_value.std())

st.markdown('* What we observe, is that, on average, **fare amounts consistently vary less than trip duration**. What this suggests is that increasing trip duration does little to increase fare amount. Hence fare amount is largely a function of trip distance rather than trip duration.')

st.write('* Hence, if the strategy was to increase fare amount, then one would have to increase distance, at the expense of making less pick-ups. Our hypothesis is that increasing the number of pick-ups is the better option. In the next section, we use KMeans clustering to see if we can determine the behaviour of high earning drivers.')

############################################################################
st.subheader('Characterising drivers by work hours and earnings')

st.markdown('**Key takeaway: KMeans appears to effectively characterise drivers by earnings and work hours.**')

cluster_agg = pd.read_csv('cluster_agg.csv')

fig = px.bar(cluster_agg, x='predicted_cluster', y=' fare_amount', 
             title="Average daily earnings (fare amount) by cluster",
             width=900,
                 height=600)

st.plotly_chart(fig)

st.write('On the dimension of average daily earnings, we see that the 3 clusters are clearly differentiated, with group 1 being the highest average earners.')

hours = ['pickup_hour_0', 'pickup_hour_1', 'pickup_hour_2',
       'pickup_hour_3', 'pickup_hour_4', 'pickup_hour_5', 'pickup_hour_6',
       'pickup_hour_7', 'pickup_hour_8', 'pickup_hour_9', 'pickup_hour_10',
       'pickup_hour_11', 'pickup_hour_12', 'pickup_hour_13', 'pickup_hour_14',
       'pickup_hour_15', 'pickup_hour_16', 'pickup_hour_17', 'pickup_hour_18',
       'pickup_hour_19', 'pickup_hour_20', 'pickup_hour_21', 'pickup_hour_22',
       'pickup_hour_23']

list_of_plots = []

for h in hours:
    list_of_plots.append(go.Bar(name=h, x=cluster_agg['predicted_cluster'], y=cluster_agg[h]))

fig = go.Figure(data=list_of_plots)
# Change the bar mode
fig.update_layout(barmode='group', 
                  title="Average number of Pick-ups per hour by cluster",
                  xaxis_title="Cluster",
                  yaxis_title="Number of Pick-ups",
                  legend_title="Hour of Day",
                  width=900,
                  height=600)

st.plotly_chart(fig)

st.write('* The 3 clusters have the following observed characteristics:')
st.markdown('**Group 0**: Lowest average daily earners, and also the lowest number of average pick-ups, and get less active later in the evening.')
st.markdown('**Group 1**: Highest average daily earners, and also the highest number of average pick-ups - most active in the late evenings.')
st.markdown('**Group 2**: Middle average daily earners, and also the middle number of average pick-ups - in between group 0 and group 1.')

st.markdown('* These observations somewhat confirm our strategy to maximise revenue: **High earners prioritise the optimisation of their number of pick-ups and work later in the evening.**')

#st.write('* Clusters 0, 1 and 2 contain 1867, 5557 and 5990 drivers respectively.')

##########################################

st.subheader('A model to predict trip fare')

st.write('In this section, we build a simple linear model using Lasso regression to see if we can predict the fare amount for a trip.')

st.markdown('**Key Takeaway: The model is not terribly accurate, but coefficients of the model seems to confirm our earlier observation that fare amounts are not highly affected by trip duration.**')

# First lets do some feature engineering to obtain avg trip and avg distance for our data
df_manhattan_stats = df_manhattan.groupby(['pickup_dropoff_man', 'pickup_hour']).agg({'trip_id': pd.Series.nunique,
                                                        'trip_in_mins': 'mean',
                                                        ' trip_distance': 'mean',
                                                        ' fare_amount': 'mean',
                                                        ' tip_amount': 'mean',
                                                        ' total_amount': 'mean'}).sort_values(by='trip_id', ascending=False)

df_manhattan_stats.reset_index(inplace=True)

df_manhattan_stats.rename(columns={'trip_id': 'number_of_trips_for_hour', 'trip_in_mins': 'avg_trip_time_mins',
                                  ' trip_distance':'avg_trip_distance', ' fare_amount':'avg_fare_amount',
                                   ' tip_amount': 'avg_tip_amount',
                                  ' total_amount': 'avg_total_amount'}, inplace=True)


df_man_predict_data = df_manhattan[['pickup_hour', ' pickup_latitude', ' pickup_longitude', 
                                    'pickup_manhattan', 'dropoff_manhattan', 'pickup_dropoff_man',
                                     ' fare_amount']]


df_man_predict_data = df_man_predict_data.merge(df_manhattan_stats[['pickup_dropoff_man', 'pickup_hour'
                                                                    , 'avg_trip_distance', 'avg_trip_time_mins']], 
                                                on=['pickup_dropoff_man', 'pickup_hour'], how='left')

with st.beta_expander("Correlations: "):
	from matplotlib import pyplot as plt

	corr = df_man_predict_data.corr()

	fig, ax = plt.subplots(figsize=(5,5))

	ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
	)
	ax.set_title('Correlation Matrix')

	ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
	);

	st.pyplot(fig)

	fig = px.scatter_3d(df_man_predict_data, x='avg_trip_distance', y='avg_trip_time_mins', z=' fare_amount',
              color= 'pickup_dropoff_man',
              title = "Fare amount vs. avg trip distance and avg trip time",
              width=800,
              height=600)

	st.plotly_chart(fig)

	st.write('We see that with the exception of the Lower Manhattan - Lower Manhattan route, it seems that the fare amount is fairly linear with these two features, particularly, with average distance. However, it is clear that average distance and average trip time are probably quite correlated, so we do need to be a bit careful about how we interpret the coefficients.')

	st.write('The regression model will probably work much better by taking out the Lower Manhattan - Lower Manhattan route. The model can also almost certainly be improved with tighter categorisation of the locations to areas of manhattan.')

##########################################
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = df_man_predict_data[['avg_trip_distance', 'avg_trip_time_mins']].values
y = df_man_predict_data[' fare_amount'].values

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.4, random_state=42)

lasso = LassoCV(cv=5, random_state=0).fit(X_train, y_train)

#print('R^2: ' + str(lasso.score(X_test, y_test)))

#print(lasso.coef_)

#print('alpha: ' + str(lasso.alpha_))

# Predict on the test data: y_pred
y_pred = lasso.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#print("Root Mean Squared Error: {}".format(rmse))
##########################################

with st.beta_expander("Lasso Regression Model details: "):

	st.write('We try to use Lasso Regression to build a predictive model for fare amount. Based on the correlation matrix above we use the following features, which seem to be most correlated to fare amount: average trip distance and average trip time in minutes. The model could certainly be improved with additional features, such as day of the week (i.e. weekend vs. week day) or weather (i.e. when it rains, trip times may increase).')

	st.write('The coefficient of determination, R^2 for the model is: ' + str(lasso.score(X_test, y_test)))

	st.write('The coefficients of average trip distance and average trip time in minutes are respectively:')

	st.write(lasso.coef_)

	st.write('These coefficients can be interpreted as follows: holding all other features equal, on average a 1 mile increase in trip distance will result in a +1.9074 in fare amount. Similarly, on average, a 1 minute increase in trip time will result in a +0.3584 in fare amount.')

	st.write('The regularization constant, alpha is: ' + str(lasso.alpha_))

	st.write("Root Mean Squared Error is: {}".format(rmse))

##########################################

st.subheader('Select Pick-up and Drop-off location to predict your fare: ')

pickup_locations = list(df_man_predict_data['pickup_manhattan'].unique())
dropoff_locations = list(df_man_predict_data['dropoff_manhattan'].unique())

#hour0 = st.slider('Select Pick-up Hour', 0, 23, 17)
pickup_selection = st.selectbox('Select pick-up location:', pickup_locations)
dropoff_selection = st.selectbox('Select drop-off location:', dropoff_locations)


avg_trip_time_mins = df_man_predict_data[(df_man_predict_data['pickup_manhattan']==pickup_selection) 
                                         & (df_man_predict_data['dropoff_manhattan']==dropoff_selection)]['avg_trip_time_mins'].values[0]

avg_trip_distance = df_man_predict_data[(df_man_predict_data['pickup_manhattan']==pickup_selection) 
                                         & (df_man_predict_data['dropoff_manhattan']==dropoff_selection)]['avg_trip_distance'].values[0]


st.write('Your predicted trip fare is: ' + str(lasso.predict([[avg_trip_distance, avg_trip_time_mins]])))

#############################################################################
st.header('Rule 3: Choose "highest value" routes')

st.write('In this last section, we develop a scoring system for routes within Manhattan which takes into account both the number of trips between locations and the average fare amount between the locations.')

# split the pick-up, drop-off into two columns again
top_10_manhattan_trips[['pickup_man','dropoff_man']] = top_10_manhattan_trips['pickup_dropoff_man'].str.split("-",expand=True,)

manhattan_areas = list(top_10_manhattan_trips['pickup_man'].unique())


manhattan_areas_lat = [40.72233902450169, 40.7549, 40.7736, 40.7870, 40.7150, 40.7580]

manhattan_areas_lon = [-74.00003896308684, -73.9840, -73.9566, -73.9754, -73.9843, -73.9855]

manhattan_areas_df = pd.DataFrame(list(zip(manhattan_areas, manhattan_areas_lat, manhattan_areas_lon)), 
               columns =['pickup_man', 'pickup_area_lat', 'pickup_area_lon'])

top_10_manhattan_trips = top_10_manhattan_trips.merge(manhattan_areas_df, on=['pickup_man'], how ="left" )


drop_areas_df = pd.DataFrame(list(zip(manhattan_areas, manhattan_areas_lat, manhattan_areas_lon)), 
               columns =['dropoff_man', 'dropoff_area_lat', 'dropoff_area_lon'])


top_10_manhattan_trips = top_10_manhattan_trips.merge(drop_areas_df, on=['dropoff_man'], how ="left" )

st.write('The "route-value" for a particular route is calculated in the following way:')

st.write('Route value = Average fare amount for the hour for the route X Average number of trips for the route for the hour')

st.write('Expand the section beneath for a description of the strategy for how a driver should maxmise their revenue.')

########################################

with st.beta_expander("How to use tool to maximise earnings."):
	st.write('* The map below, shows trips within Manhattan for each hour. To each route, we associate a "route value" (displayed in brackets in the legend next to each route). This value is calculated as above.')

	st.write('* The strategy is then as follows: A driver should at each hour (and location) select the highest value route from their current location and try to obtain a trip for their route. For example, at 8am, the highest value route is (midtown-midtown), then at 9am, from midtown, the highest value route is again (midtown-midtown) and again at 10, 11 and 12. Finally at 1pm, the highest value route is midtown-Lower-Manhattan, so the driver should select this route.')
 	
	st.write('I think what we have done here is similar to a Markov chain.')

hour = st.slider('Select Hour', 0, 23, 17)

route_list = list(top_10_manhattan_trips[top_10_manhattan_trips['pickup_hour']==hour]['pickup_dropoff_man'])

temp_df = top_10_manhattan_trips[top_10_manhattan_trips['pickup_hour']==hour]


fig = go.Figure()

for route in route_list:
    fig.add_trace(go.Scattermapbox(
        mode = "markers+lines",
        lon = [temp_df[temp_df['pickup_dropoff_man']==route]['pickup_area_lon'].values[0], temp_df[temp_df['pickup_dropoff_man']==route]['dropoff_area_lon'].values[0]],
        lat = [temp_df[temp_df['pickup_dropoff_man']==route]['pickup_area_lat'].values[0], temp_df[temp_df['pickup_dropoff_man']==route]['dropoff_area_lat'].values[0]],
        marker = {'size': 20},
        line = {'width': 5},
        hovertemplate = temp_df[temp_df['pickup_dropoff_man']==route]['route_value_for_hour'].values[0],
        name = route + ' ' + '(' + str(temp_df[temp_df['pickup_dropoff_man']==route]['route_value_for_hour'].values[0]) +')')
    	)

fig.update_layout(
    title = "Most valuable routes for hour: " + str(hour),
    margin ={'l':0,'t':0,'b':0,'r':0},
    mapbox = {
        'center': {'lon': -73.98488905234063, 'lat': 40.76263444496066},
        'style': "stamen-terrain",
        'zoom': 11},
        width = 1000,
        height = 800,
        legend=dict(
    		yanchor="top",
    		y=0.99,
    		xanchor="left",
    		x=0.01)
)

st.plotly_chart(fig)

with st.beta_expander("Comments on maximising revenue for a company of 10 Taxi-Drivers"):
	st.write('* Our strategy to maximise revenue for one driver could essentially be extended to a team of 10 drivers. Each driver should concentrate on the Manhattan area (particularly Lower-Manhattan and Midtown) and decide on passengers by picking the highest valued route using the tool above.')

	st.write('* Potential data issues affecting the approach: The categorisation of pick-up and drop-off locations were done fairly arbitrarily (and manually) and thus may not be 100% accurate. It could certainly be improved with better knowledge of New York City and a more informed allocation of pick-up and drop-off locations to districts of Manhattan. Obviously, accuracy of the dataset could also always be an issue.')

	st.write('*  Potential drawbacks of approach: The strategy does not take into account the physical toll that could be imposed on the drivers (continually trying to optimise number of pick-ups). An alternative approach may be to mix short trips, and some higher value longer trips out of Manhattan, say to Laguardia or JFK airports. There also could be a better formula for the "route vale".')

with st.beta_expander("Technical notes:"):
	st.write('* For practicality, the analysis and modelling completed here was based on a sample of 30,000 trips in the month of August (which contained 12 million + trips).')

	st.write('* The sample was statistically validated to reflect the same distribution as the complete dataset for August 2013, by comparing the distributions (and performing the Kolmogorov-Smirnov test for categorical features and two sample Chi Squared for continuous features) of each major feature: fare amounts, passenger counts, trip times, trip distances, locations etc.')

	st.write('* Analysis and dashboard completed using Python and Streamlit!')

#############################################################################








