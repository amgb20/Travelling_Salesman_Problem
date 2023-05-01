import React, { useState, useCallback, useRef } from 'react';
import { GoogleMap, LoadScript, useLoadScript } from '@react-google-maps/api';
import './MapComponent.css';


const MapComponent = () => {
  const [rectangle, setRectangle] = useState(null);
  const [isDrawingMode, setIsDrawingMode] = useState(false);
  const mapRef = useRef();
  const drawingManagerRef = useRef();

  const mapStyles = {
    height: '100%',
    width: '100%',
  };

  const defaultCenter = {
    lat: -27.41613283804977,
    lng: 21.683633361620455,
  };

  const rectangleRef = useRef();

  const onRectangleComplete = (rectangle) => {
    if (rectangleRef.current) {
      rectangleRef.current.setMap(null);
    }
    rectangleRef.current = rectangle;
  };


  const handleDrawingModeClick = () => {
    setIsDrawingMode((prevState) => !prevState);
    drawingManagerRef.current.setOptions({
      drawingMode: isDrawingMode ? null : window.google.maps.drawing.OverlayType.RECTANGLE,
    });
  };

  const handleButtonClick = () => {
    if (!rectangleRef.current) {
      alert('Please draw a rectangle on the map first.');
      return;
    }
    const bounds = rectangleRef.current.getBounds();
    const ne = bounds.getNorthEast();
    const sw = bounds.getSouthWest();
    const nw = new window.google.maps.LatLng(ne.lat(), sw.lng());
    const se = new window.google.maps.LatLng(sw.lat(), ne.lng());

    const coordinates = `The GPS coordinates of the rectangle corners are:
      NE: ${ne.lat()}, ${ne.lng()}
      NW: ${nw.lat()}, ${nw.lng()}
      SW: ${sw.lat()}, ${sw.lng()}
      SE: ${se.lat()}, ${se.lng()}`;

    document.getElementById('coordinatesDisplay').value = coordinates;
  };


  const onMapLoad = useCallback((map) => {
    mapRef.current = map;
    const drawingManager = new window.google.maps.drawing.DrawingManager({
      drawingMode: window.google.maps.drawing.OverlayType.RECTANGLE,
      drawingControl: false,
      rectangleOptions: {
        draggable: true,
        editable: true,
        zIndex: 1,
      },
    });
    drawingManager.setMap(map);
    drawingManagerRef.current = drawingManager;
    window.google.maps.event.addListener(drawingManager, 'rectanglecomplete', onRectangleComplete);
  }, [onRectangleComplete]);

  const { isLoaded } = useLoadScript({
    googleMapsApiKey: 'AIzaSyDB6rg5lFeylHpR2NHnP0PdgXPoIiwslc4',
    libraries: ['drawing'],
  });

  // return (
  //   <div className="map-container">
  //     {isLoaded ? (
  //       <>
  //         <div className="controls-container">
  //           <button className="button" onClick={handleDrawingModeClick}>
  //             {isDrawingMode ? 'Stop Drawing' : 'Start Drawing'}
  //           </button>
  //           <button className="button" onClick={handleButtonClick}>
  //             Submit
  //           </button>
  //           <textarea id="coordinatesDisplay" readOnly className="coordinates-display" />
  //         </div>
  //         <GoogleMap
  //           mapContainerStyle={mapStyles}
  //           zoom={13}
  //           center={defaultCenter}
  //           onLoad={onMapLoad}
  //         />
  //       </>
  //     ) : (
  //       <div>Loading...</div>
  //     )}
  //   </div>
  // );
  return (
    <div>
      {isLoaded ? (
        <>
          <div className="controls-container">
            <button className="button" onClick={handleDrawingModeClick}>
              {isDrawingMode ? 'Stop Drawing' : 'Start Drawing'}
            </button>
            <button className="button" onClick={handleButtonClick}>
              Submit
            </button>
            <div className='print-box'>
              <textarea id="coordinatesDisplay" readOnly className="coordinates-display" />
            </div>
          </div>
          <div className="map-container">
            <GoogleMap
              mapContainerStyle={mapStyles}
              zoom={13}
              center={defaultCenter}
              onLoad={onMapLoad}
            />
          </div>
        </>
      ) : (
        <div>Loading...</div>
      )}
    </div>
  );

};

export default MapComponent;
