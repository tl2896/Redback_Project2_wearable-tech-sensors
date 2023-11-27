import 'dart:async';

import 'package:flutter/material.dart';
import 'package:location/location.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'package:flutter_polyline_points/flutter_polyline_points.dart';

class MapPage extends StatefulWidget {
  MapPage({super.key});
  bool _isTracking = false;
  void startStopTracking(bool isTracking) {
    _isTracking = isTracking;
  }

  List<LatLng> _polylineCoords = [];
  List<LatLng> getPolyList() {
    return _polylineCoords;
  }

  @override
  State<MapPage> createState() => _MapPageState();
}

class _MapPageState extends State<MapPage> {
  final Completer<GoogleMapController> _mapController =
      Completer<GoogleMapController>();
  Location _locationController = new Location();
  LatLng? _currentPosition = null;
  bool _isTracking = true;

  List<LatLng> _coordsList = [];

  Polyline _polyLine = Polyline(
    polylineId: PolylineId("userRoute"),
    color: Colors.blue,
    width: 5,
    points: [],
  );

  // test variables
  static const LatLng _initPos = LatLng(-38.1986127, 144.2986353);
  static const LatLng _endPos = LatLng(-38.1990677, 144.3081407);

  @override
  void initState() {
    super.initState();
    widget._polylineCoords = [];
    setLocation();
    //Timer.periodic(Duration(seconds: 10), (timer) {
    //getLocationUpdates();
    //});
  }

  @override
  void dispose() {
    //_mapController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _currentPosition == null
          ? Center(
              child: Text("Loading..."),
            )
          : GoogleMap(
              onMapCreated: ((GoogleMapController controller) =>
                  _mapController.complete(controller)),
              initialCameraPosition: CameraPosition(
                target: _initPos,
                zoom: 15,
              ),
              markers: {
                // Marker(
                //     markerId: MarkerId("_startLocation"),
                //     icon: BitmapDescriptor.defaultMarker,
                //     position: _currentPosition!),
                Marker(
                    markerId: MarkerId("_currentLocation"),
                    icon: BitmapDescriptor.defaultMarker,
                    position: _currentPosition!)
              },
              polylines: {_polyLine},
            ),
    );
  }

  Future<void> cameraFollow(LatLng pos) async {
    GoogleMapController controller = await _mapController.future;
    CameraPosition newCameraPosition = CameraPosition(
      target: pos,
      zoom: 15,
    );
    await controller.animateCamera(
      CameraUpdate.newCameraPosition(newCameraPosition),
    );
  }

  Future<void> setLocation() async {
    // Begin checking location services available and user has permission
    bool serviceEnabled;
    PermissionStatus permissionGranted;

    serviceEnabled = await _locationController.serviceEnabled();
    if (serviceEnabled) {
      serviceEnabled = await _locationController.requestService();
    } else {
      return;
    }

    permissionGranted = await _locationController.hasPermission();
    // end permission and service checks
    if (permissionGranted == PermissionStatus.denied) {
      permissionGranted = await _locationController.requestPermission();

      if (permissionGranted != PermissionStatus.granted) {
        return;
      }
    }

    // _locationController.onLocationChanged
    //     .listen((LocationData currentLocation) {

    LocationData currentLocation = await _locationController.getLocation();

    if (currentLocation.latitude != null && currentLocation.longitude != null) {
      setState(() {
        _currentPosition =
            //TODO change this before prduction
            // actual code
            //LatLng(currentLocation.latitude!, currentLocation.longitude!);

            // test code
            _initPos;
      });

      // Add polyline co-ord
      _coordsList.add(_currentPosition!);
      widget._polylineCoords = _coordsList;
      print("Polyline coords: ${widget._polylineCoords}");
      cameraFollow(_currentPosition!);
      int timercount = 0;
      Timer.periodic(Duration(seconds: 10), (timer) {
        getLocationUpdates();
        timercount++;
        if (timercount >= 1) {
          timer.cancel();
        }
      });
    }

    //});
  } // end getLocationUpdates

  Future<void> getLocationUpdates() async {
    // Begin checking location services available and user has permission
    bool serviceEnabled;
    PermissionStatus permissionGranted;

    serviceEnabled = await _locationController.serviceEnabled();
    if (serviceEnabled) {
      serviceEnabled = await _locationController.requestService();
    } else {
      return;
    }

    permissionGranted = await _locationController.hasPermission();
    // end permission and service checks
    if (permissionGranted == PermissionStatus.denied) {
      permissionGranted = await _locationController.requestPermission();

      if (permissionGranted != PermissionStatus.granted) {
        return;
      }
    }

    // _locationController.onLocationChanged
    //     .listen((LocationData currentLocation) {
    if (_isTracking) {
      //**********Actual code
      //TODO use this for production */
      // LocationData currentLocation = await _locationController.getLocation();
      // if (currentLocation.latitude != null &&
      //     currentLocation.longitude != null) {
      //   setState(() {
      //     _currentPosition =
      //         LatLng(currentLocation.latitude!, currentLocation.longitude!);

      LocationData currentLocation = await _locationController.getLocation();
      if (currentLocation.latitude != null &&
          currentLocation.longitude != null) {
        setState(() {
          _currentPosition = _endPos;
        });

        // Add polyline co-ord
        _coordsList.add(_currentPosition!);
        widget._polylineCoords = _coordsList;
        // update polylines
        _polyLine = Polyline(
            polylineId: PolylineId("userRoute"),
            color: Colors.blue,
            width: 5,
            points: widget._polylineCoords);

        // camera to follow
        cameraFollow(_currentPosition!);
      }
    }
    //});
  } // end getLocationUpdates

  // **** poly line and waypoints

  // ******* Logging functions

  double? distanceCovered = null;
  double? workOutSeconds = null;
  double? evelvationChange = null;
  double? avgSpeed = null;
}// end class


