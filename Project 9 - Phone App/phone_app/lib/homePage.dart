import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

import 'Friends.dart';
import 'MyActivity.dart';
import 'main.dart';
import 'myAccount.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: HomePage(title: "Home Page"),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({Key? key, required this.title}) : super(key: key);
  final String title;

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _currentIndex = 0;

  @override
  double totalAvgPace = 32;
  double totalKilometer = 100;
  double totalTime = 7;
  double totalAvgTime = 10;
  int totalWorkouts = 0;
  void initState() {
    super.initState();
    retrieveLogs();
  }

  Widget build(BuildContext context) {
    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          backgroundColor: const Color(0xff87A395),
          flexibleSpace: const Column(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              TabBar(
                tabs: [
                  Tab(
                    text: 'Home',
                  ),
                  Tab(
                    text: 'Activities',
                  ),
                  Tab(
                    text: 'Account',
                  ),
                ],
              ),
            ],
          ),
        ),
        body: TabBarView(
          children: [
            FutureBuilder<void>(
              future: retrieveLogs(),
              builder: (context, snapshot) {
                if (snapshot.connectionState == ConnectionState.waiting) {
                  return Center(child: CircularProgressIndicator());
                } else if (snapshot.hasError) {
                  return Center(child: Text('Error: ${snapshot.error}'));
                } else {
                  return HomeTab(
                    totalKilometer: totalKilometer,
                    totalTime: totalTime,
                    avgTime: totalAvgTime,
                    avgPace: totalAvgPace,
                    workouts: totalWorkouts,
                  );
                }
              },
            ),
            // HomeTab(
            //   totalKilometer: totalKilometer,
            //   totalTime: totalTime,
            //   avgTime: totalAvgTime,
            //   avgPace: totalAvgPace,
            // ),
            MyActivity(
              title: '',
            ),
            MyAccount(title: ''),
          ],
        ),
        bottomNavigationBar: BottomNavigationBar(
          currentIndex: _currentIndex,
          onTap: (index) {
            setState(() {
              _currentIndex = index;
              switch (_currentIndex) {
                case 1:
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => const MyFriendScreen(title: ''),
                    ),
                  );
                  break;
                case 2:
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => Setting(title: "Settings"),
                    ),
                  );
                  break;
              }
            });
          },
          items: const [
            BottomNavigationBarItem(
              icon: Icon(Icons.home),
              label: 'Home',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.group),
              label: 'Friends',
            ),
            BottomNavigationBarItem(
              icon: Icon(Icons.settings),
              label: 'Settings',
            ),
          ],
        ),
      ),
    );
  }

  Future<void> retrieveLogs() async {
    String? userID = FirebaseAuth.instance.currentUser?.uid.toString();

    if (userID == null) {
      return;
    }

    FirebaseFirestore db = FirebaseFirestore.instance;

    final userWorkoutRef =
        db.collection("users").doc(userID).collection('workouts');
    QuerySnapshot snapShot = await userWorkoutRef.get();
    final data = snapShot.docs.map((doc) => doc.data()).toList();
    calculateTotals(data);
  }

  void calculateTotals(List data) {
    double distance = 0;
    double time = 0;
    int counter = 0;
    double avgPace = 0;
    double avgTime = 0;
    for (var workout in data) {
      counter++;
      distance += workout['distance'];
      time += workout['time'];
      avgPace += double.parse(workout["avgPace"]);
    }
    totalKilometer = distance;
    totalTime = time;
    totalAvgPace = avgPace / counter;
    totalAvgTime = totalTime / counter;
    totalWorkouts = counter;
  }
}

class HomeTab extends StatelessWidget {
  final double totalKilometer;
  final double totalTime;
  final double avgTime;
  final double avgPace;
  final int workouts;

  HomeTab({
    required this.totalKilometer,
    required this.totalTime,
    required this.avgTime,
    required this.avgPace,
    required this.workouts,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Color(0xFF8F9E91),
      child: Stack(
        children: [
          Column(
            children: [
              SizedBox(height: 40),
              SizedBox(height: 20),
              Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Padding(
                      padding: const EdgeInsets.only(top: 500),
                      child: Container(
                        width: 300,
                        child: ElevatedButton(
                          onPressed: () {},
                          style: ElevatedButton.styleFrom(
                            primary: Colors.black,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(8.0),
                            ),
                          ),
                          child: const Text(
                            "START",
                            style: TextStyle(
                              color: Colors.white,
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          Positioned(
            top: 80,
            left: 20,
            right: 20,
            child: SizedBox(
              height: 450,
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
                            '${totalKilometer.toStringAsFixed(2)} Km',
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
                            'Total Distance',
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
                                      _formatTime(totalTime.toInt()),
                                      style: const TextStyle(
                                        fontSize: 24,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white,
                                      ),
                                    ),
                                  ),
                                  Text(
                                    'Total Time',
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
                                      '$workouts',
                                      style: TextStyle(
                                        fontSize: 24,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white,
                                      ),
                                    ),
                                  ),
                                  Text(
                                    'Workouts',
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
                                      _formatTime((avgTime).toInt()),
                                      style: TextStyle(
                                        fontSize: 24,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white,
                                      ),
                                    ),
                                  ),
                                  Text(
                                    'Average Time',
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
                                      '${avgPace.toStringAsFixed(2)}',
                                      style: TextStyle(
                                        fontSize: 24,
                                        fontWeight: FontWeight.bold,
                                        color: Colors.white,
                                      ),
                                    ),
                                  ),
                                  Text(
                                    'Avg Pace (km/h)',
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
            ),
          ),
        ],
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
}

class ProfileTab extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Text('Profile Content'),
    );
  }
}
