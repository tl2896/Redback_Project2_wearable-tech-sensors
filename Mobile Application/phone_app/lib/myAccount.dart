import 'dart:math';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'main.dart';
import 'privacy.dart';
import 'editProfile.dart';
//import 'myActivity.dart';
//import 'main.dart';
//import 'homePage.dart';

class MyAccount extends StatefulWidget {
  const MyAccount({super.key, required this.title});
  final String title;

  @override
  State<MyAccount> createState() => _MyAccount();
}

class _MyAccount extends State<MyAccount> {
  //int _currentIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          Container(
            color: const Color(0xFF8F9E91),
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Transform.translate(
                    offset: const Offset(0.0, -260.0),
                    child: const Text(
                      "Accounts",
                      style: TextStyle(
                        fontFamily: 'Roboto',
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                  const SizedBox(height: 50),
                ],
              ),
            ),
          ),
          Positioned(
            left: 50,
            bottom: 450,
            child: Container(
              width: 300,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) =>
                          EditProfileActivity(title: 'Profile'),
                    ),
                  );
                },
                style: ElevatedButton.styleFrom(
                  primary: Colors.transparent,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                child: const Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: [
                        Icon(
                          Icons.account_circle,
                          color: Colors.white,
                        ),
                        SizedBox(width: 10),
                        Text(
                          "Edit Profile",
                          style: TextStyle(fontSize: 18, color: Colors.white),
                        ),
                      ],
                    ),
                    Icon(
                      Icons.arrow_forward,
                      color: Colors.white,
                    ),
                  ],
                ),
              ),
            ),
          ),
          Positioned(
            left: 50,
            bottom: 390,
            child: Container(
              width: 300,
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => Privacy(),
                    ),
                  );
                },
                style: ElevatedButton.styleFrom(
                  primary: Colors.transparent,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                child: const Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: [
                        Icon(
                          Icons.privacy_tip,
                          color: Colors.white,
                        ),
                        SizedBox(width: 10),
                        Text(
                          "Privacy",
                          style: TextStyle(fontSize: 18, color: Colors.white),
                        ),
                      ],
                    ),
                    Icon(
                      Icons.arrow_forward,
                      color: Colors.white,
                    ),
                  ],
                ),
              ),
            ),
          ),
          Positioned(
            left: 50,
            bottom: 330,
            child: Container(
              width: 300,
              child: ElevatedButton(
                onPressed: () {},
                style: ElevatedButton.styleFrom(
                  primary: Colors.transparent,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                child: const Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: [
                        Icon(
                          Icons.email,
                          color: Colors.white,
                        ),
                        SizedBox(width: 10),
                        Text(
                          "Email Address",
                          style: TextStyle(fontSize: 18, color: Colors.white),
                        ),
                      ],
                    ),
                    Icon(
                      Icons.arrow_forward,
                      color: Colors.white,
                    ),
                  ],
                ),
              ),
            ),
          ),
          Positioned(
            left: 50,
            bottom: 270,
            child: Container(
              width: 300,
              child: ElevatedButton(
                onPressed: () {
                  logOut();
                },
                style: ElevatedButton.styleFrom(
                  primary: Colors.red,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(10),
                  ),
                ),
                child: const Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(
                      Icons.cancel,
                      color: Colors.white,
                    ),
                    SizedBox(width: 10),
                    Text(
                      "LOGOUT",
                      style: TextStyle(fontSize: 18, color: Colors.white),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
      // bottomNavigationBar: BottomNavigationBar(
      //   currentIndex: _currentIndex,
      //   onTap: (index) {
      //     setState(() {
      //       _currentIndex = index;
      //       switch (_currentIndex) {
      //         case 0:
      //           Navigator.push(
      //             context,
      //             MaterialPageRoute(
      //               builder: (context) => HomePage(title: "HomePage"),
      //             ),
      //           );
      //           break;
      //         case 1:
      //           Navigator.push(
      //             context,
      //             MaterialPageRoute(
      //               builder: (context) => MyActivity(title: "MyActivity"),
      //             ),
      //           );
      //           break;
      //         case 2:
      //           Navigator.push(
      //             context,
      //             MaterialPageRoute(
      //               builder: (context) => Setting(title: "MyHomePage"),
      //             ),
      //           );
      //           break;
      //       }
      //     });
      //   },
      //   items: [
      //     BottomNavigationBarItem(
      //       icon: Icon(Icons.home),
      //       label: 'Home',
      //     ),
      //     BottomNavigationBarItem(
      //       icon: Icon(Icons.accessibility),
      //       label: 'Activities',
      //     ),
      //     BottomNavigationBarItem(
      //       icon: Icon(Icons.settings),
      //       label: 'Settings',
      //     ),
      //   ],
      // ),
    );
  }

  Future<void> logOut() async {
    await FirebaseAuth.instance.signOut();
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => MyApp()),
    );
  }
}
