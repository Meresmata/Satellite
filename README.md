# Satellite
Use Sentinel-5 Satellite data to extract environmental data. Show the region intensity and the time trend in the data. 
For the classification of the areas use the land cover map of https://lcviewer.vito.be/.

Use rural and urban area as classes. As most people are living in villages and cities were the gases take larger effect,
because

1. this is are the places where the gases are emitted 
2. thus the concentration should be normally higher
3. this is the more people are living in the area and can be effected

For the polygon of a nation the shape from geopandas are used. But also the polygons extracted from open street map should 
be fine.
 
## Procedure:
1. Download the data for a nation for a given period of time. 
2. Remove data outside the nation.
3. Reduce data size with H3.
4. Use land cover map of the copernicus project. To differentiate between rural and urban areas.
5. Create trend and map for the nation.