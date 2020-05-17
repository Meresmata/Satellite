# Satellite
Use Sentinel-5 Satellite data to extract environmental data. 
Use Sentinel-2 (resolution 10m) data to for classification of the data points of Sentinel-5 data as rural or urban area.


# Classification 214 x 214 pixels of the network
1. rural: snow, desert, fields, forests
2. urban: industrial area, or at least approx. 25% living space

# Indirect classes (not tested separately before the network)
3. error: black, or mostly black rasters
4. others: exclude rasters from network, that are not part of the country, as other countries or open sea
5. mixed: optional, several networks classified differently

Create a classification of every coordinate (in respect to a set spacial resolution).
This classification shall be able to be used for the map creation. Should be able to 
be taken into account when further graphs are being create.  
