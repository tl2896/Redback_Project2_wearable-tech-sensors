import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';
import 'dart:async';
import 'gps.dart';
import 'results.dart';
import 'seconds_singleton.dart';

class TimerPage extends StatefulWidget {
  final String title;
  TimerPage({required this.title});
  int seconds = 0;
  int elapsed = 0;
  int getSeconds() {
    return elapsed;
  }

  @override
  _TimerPageState createState() => _TimerPageState();
}

class _TimerPageState extends State<TimerPage> {
  int _seconds = 0;
  late Timer _timer;
  bool _isPaused = false;
  bool _isRunning = false;

  @override
  void initState() {
    super.initState();
    _timer = Timer.periodic(Duration(seconds: 1), _incrementTimer);
  }

  void _incrementTimer(Timer timer) {
    if (!_isPaused && _isRunning) {
      setState(() {
        _seconds++;
        TimerSingleton().seconds = _seconds;
        widget.seconds = _seconds;
      });
    }
  }

  String _formatTime(int seconds) {
    int minutes = seconds ~/ 60;
    int remainingSeconds = seconds % 60;
    String minutesStr = minutes.toString().padLeft(2, '0');
    String secondsStr = remainingSeconds.toString().padLeft(2, '0');
    return '$minutesStr:$secondsStr';
  }

  void _togglePause() {
    setState(() {
      _isPaused = !_isPaused;
    });
  }

  void _startStopTimer() {
    if (!_isRunning) {
      _isRunning = true;
      MapPageSingleton().mapPageInstance.startStopTracking(true);
    } else {
      _timer.cancel();
      MapPageSingleton().mapPageInstance.startStopTracking(false);
      List<LatLng> polyPoints =
          MapPageSingleton().mapPageInstance.getPolyList();
      Navigator.of(context).push(MaterialPageRoute(
          builder: (context) => Results(
                polyLinePoints: polyPoints,
                seconds: widget.elapsed,
              )));
    }
  }

  @override
  void dispose() {
    widget.elapsed = _seconds;
    _timer.cancel();
    MapPageSingleton().mapPageInstance.startStopTracking(false);
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        color: const Color(0xFF8F9E91),
        child: ListView(
          padding: EdgeInsets.all(16.0),
          children: [
            Text(
              widget.title,
              style: TextStyle(
                fontSize: 40,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 20),
            Container(
              width: 150,
              height: 150,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.transparent,
                border: Border.all(
                  color: Colors.black,
                  width: 2.0,
                ),
              ),
              child: Center(
                child: Text(
                  _formatTime(_seconds),
                  style: TextStyle(
                    fontSize: 40,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
              ),
            ), // container holding circle elapsed time display
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _togglePause,
              style: ElevatedButton.styleFrom(
                primary: Colors.black,
                minimumSize: Size(160, 60),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              child: Text(_isPaused ? 'Resume' : 'Pause'),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _startStopTimer,
              style: ElevatedButton.styleFrom(
                primary: Colors.red,
                minimumSize: Size(160, 60),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
              child: Text(_isRunning ? 'Stop' : 'Start'),
            ),
            SizedBox(height: 50),
            Container(
              width: MediaQuery.of(context).size.width,
              constraints: BoxConstraints(
                maxWidth: MediaQuery.of(context).size.width,
                maxHeight: MediaQuery.of(context).size.width,
              ),
              child: MapPageSingleton().mapPageInstance,
            )
          ], // children of list view
        ),
      ), // body container
    );
  }
}

class TimerPageSingleton {
  static late final TimerPageSingleton _singleton =
      TimerPageSingleton._internal();

  late TimerPage timerPageInstance;

  factory TimerPageSingleton() {
    return _singleton;
  }

  TimerPageSingleton._internal() {
    timerPageInstance = TimerPage(
      title: '',
    );
  }

  void updateTitle(String newTitle) {
    timerPageInstance = TimerPage(title: newTitle);
  }
}
