import 'dart:math';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter_polyline_points/flutter_polyline_points.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'seconds_singleton.dart';

class Results extends StatefulWidget {
  //Results({super.key});
  List<LatLng> polyLinePoints;
  int seconds;
  Results({required this.polyLinePoints, super.key, required this.seconds});

  @override
  State<Results> createState() => _Results();
}

class _Results extends State<Results> {
  double totalDistance = 0;
  int totalTime = 180;
  FirebaseFirestore db = FirebaseFirestore.instance;
  @override
  void initState() {
    super.initState();
    totalDistance = calculateDistance(widget.polyLinePoints);
    totalTime = TimerSingleton().seconds;
    logData();
  }

  double calculateDistance(List<LatLng> polyLinePoints) {
    for (int i = 0; i < polyLinePoints.length - 1; i++) {
      LatLng point1 = polyLinePoints[i];
      LatLng point2 = polyLinePoints[i + 1];

      double distance = _haversineDistance(
        point1.latitude,
        point1.longitude,
        point2.latitude,
        point2.longitude,
      );

      totalDistance += distance;
    }

    return totalDistance;
  }

// Haversine formula to calculate distance
  double _haversineDistance(
    double lat1,
    double lon1,
    double lat2,
    double lon2,
  ) {
    const R = 6371.0; // Earth radius in kilometers

    double dLat = _degreesToRadians(lat2 - lat1);
    double dLon = _degreesToRadians(lon2 - lon1);

    double a = pow(sin(dLat / 2), 2) +
        cos(_degreesToRadians(lat1)) *
            cos(_degreesToRadians(lat2)) *
            pow(sin(dLon / 2), 2);

    double c = 2 * atan2(sqrt(a), sqrt(1 - a));

    return R * c; // Distance in kilometers
  }

  double _degreesToRadians(double degrees) {
    return degrees * (pi / 180.0);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Color(0xFF8F9E91),
      child: Container(
        color: Colors.brown,
        padding: EdgeInsets.all(20),
        child: Column(
          children: [
            Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                SizedBox(width: 10),
                Padding(
                  padding: EdgeInsets.all(10.0),
                  child: Text(
// FIX -----
                    '${totalDistance.toStringAsFixed(2)}',
                    style: TextStyle(
                      fontSize: 30,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
                Padding(
                  padding: EdgeInsets.only(bottom: 20.0),
                  child: Text(
                    'Total Kilometers',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
              ],
            ),
            SizedBox(height: 20),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Column(
                  children: [
                    Container(
                      width: 150,
                      height: 110,
                      color: Colors.green,
                      padding: EdgeInsets.all(10),
                      child: Column(
                        children: [
                          Padding(
                            padding: EdgeInsets.all(8.0),
                            child: Text(
// FIX ---------
                              _formatTime(totalTime),
                              style: const TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                          ),
                          Text(
                            'Total Time', //   FIX *******************************
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.white,
                            ),
                          ),
                        ],
                      ),
                    ),
                    SizedBox(height: 10),
                    Container(
                      width: 150,
                      height: 110,
                      color: Colors.green,
                      padding: EdgeInsets.all(10),
                      child: Column(
                        children: [
                          Padding(
                            padding: EdgeInsets.all(8.0),
                            child: Text(
// FIX ------------
                              'Y T I', // FIX  ******************************
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                          ),
                          Text(
                            'Y T I',
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.white,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
                Column(
                  children: [
                    Container(
                      width: 150,
                      height: 110,
                      color: Colors.green,
                      padding: EdgeInsets.all(10),
                      child: Column(
                        children: [
                          Padding(
                            padding: EdgeInsets.all(8.0),
                            child: Text(
// FIX ------
                              _avgPage(totalDistance, totalTime),
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                          ),
                          Text(
                            'Average Pace',
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.white,
                            ),
                          ),
                        ],
                      ),
                    ),
                    SizedBox(height: 10),
                    Container(
                      width: 150,
                      height: 110,
                      color: Colors.green,
                      padding: EdgeInsets.all(10),
                      child: Column(
                        children: [
                          Padding(
                            padding: EdgeInsets.all(8.0),
                            child: Text(
// FIX -----------
                              "Y T I",
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                          ),
                          Text(
                            'Y T I',
                            style: TextStyle(
                              fontSize: 16,
                              color: Colors.white,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  String _formatTime(int seconds) {
    int minutes = seconds ~/ 60;
    int remainingSeconds = seconds % 60;
    String minutesStr = minutes.toString().padLeft(2, '0');
    String secondsStr = remainingSeconds.toString().padLeft(2, '0');
    return '$minutesStr:$secondsStr';
  }

  String _avgPage(double distance, int seconds) {
    double hours = seconds / 60 / 60;
    double pace = distance / hours;
    return pace.toStringAsFixed(2);
  }

  Future<void> logData() async {
    List<Map<String, double>> latLngMapList = widget.polyLinePoints
        .map((latLng) =>
            {'latitude': latLng.latitude, 'longitude': latLng.longitude})
        .toList();

    final workOut = <String, dynamic>{
      "date": DateTime.now().toString(),
      "distance": totalDistance,
      "time": totalTime,
      "avgPace": _avgPage(totalDistance, totalTime),
      "path": latLngMapList,
    };

    try {
      String? id = FirebaseAuth.instance.currentUser?.uid;

      db.collection("users").doc(id).collection("workouts").add(workOut);
    } catch (e) {
      print("Error: " + e.toString());
    }
  }
}
