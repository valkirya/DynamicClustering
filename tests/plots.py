 #       for _, row in data.iterrows():
   
        #     if row['CLS'] == -1:
        #         cluster_colour = '#000000'
        #     else:
        #         cluster_colour = colors[row['CLS']]
    
        #     folium.CircleMarker(
        #         location= [row['LAT'], row['LON']],
        #         radius=5,
        #         popup= row['CLS'],
        #         color=cluster_colour,
        #         fill=True,
        #         fill_color=cluster_colour
        #     ).add_to(m)   