import 'package:awesome_dialog/awesome_dialog.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'homePage.dart';

void main() {
  runApp(const Login());
}

class Login extends StatelessWidget {
  const Login({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: DefaultTabController(
        length: 2,
        child: Scaffold(
          appBar: AppBar(
            title: Text('Login & Signup'),
            bottom: TabBar(
              tabs: [
                Tab(text: 'Login'),
                Tab(text: 'Signup'),
              ],
            ),
          ),
          body: TabBarView(
            children: [
              LoginCard(),
              SignupCard(),
            ],
          ),
        ),
      ),
    );
  }
}

class LoginCard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Card(
        margin: EdgeInsets.all(20.0),
        child: Padding(
          padding: EdgeInsets.all(16.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              TextField(
                decoration: InputDecoration(labelText: 'Email'),
              ),
              SizedBox(height: 10),
              TextField(
                decoration: InputDecoration(labelText: 'Password'),
                obscureText: true,
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  // Handle login logic here
                },
                child: Text('Login'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class SignupCard extends StatelessWidget {
  TextEditingController nameController = TextEditingController();
  TextEditingController emailController = TextEditingController();
  TextEditingController passwordController = TextEditingController();
  TextEditingController confirmPasswordController = TextEditingController();
  @override
  Widget build(BuildContext context) {
    return Center(
      child: Card(
        margin: EdgeInsets.all(20.0),
        child: Padding(
          padding: EdgeInsets.all(16.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: <Widget>[
              TextField(
                controller: nameController,
                decoration: InputDecoration(labelText: 'Full Name'),
              ),
              SizedBox(height: 10),
              TextField(
                controller: emailController,
                decoration: InputDecoration(labelText: 'Email'),
              ),
              SizedBox(height: 10),
              TextField(
                controller: passwordController,
                decoration: InputDecoration(labelText: 'Password'),
                obscureText: true,
              ),
              SizedBox(height: 10),
              TextField(
                controller: confirmPasswordController,
                decoration: InputDecoration(labelText: 'Confirm Password'),
                obscureText: true,
              ),
              SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  signUp(context);
                },
                child: Text('Signup'),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // Login function
  void login() {}

  // Sign up function
  Future<void> signUp(BuildContext context) async {
    UserCredential? userCredential;
    if (nameController.text.isEmpty) {
      AwesomeDialog(
              context: context,
              dialogType: DialogType.error,
              animType: AnimType.scale,
              showCloseIcon: true,
              title: "Error",
              desc: "You have not entered a your Name",
              btnCancelOnPress: () {},
              btnOkOnPress: () {})
          .show();
      return;
    }
    if (emailController.text.isEmpty) {
      AwesomeDialog(
              context: context,
              dialogType: DialogType.error,
              animType: AnimType.scale,
              showCloseIcon: true,
              title: "Error",
              desc: "You have not entered a your Email",
              btnCancelOnPress: () {},
              btnOkOnPress: () {})
          .show();
      return;
    }
    if (passwordController.text.isEmpty) {
      AwesomeDialog(
              context: context,
              dialogType: DialogType.error,
              animType: AnimType.scale,
              showCloseIcon: true,
              title: "Error",
              desc: "You have not entered a password",
              btnCancelOnPress: () {},
              btnOkOnPress: () {})
          .show();
      return;
    }
    if (confirmPasswordController.text.isEmpty) {
      AwesomeDialog(
              context: context,
              dialogType: DialogType.error,
              animType: AnimType.scale,
              showCloseIcon: true,
              title: "Error",
              desc: "You have not confirmed your password",
              btnCancelOnPress: () {},
              btnOkOnPress: () {})
          .show();
      return;
    }

    if (confirmPasswordController.text.compareTo(passwordController.text) !=
        0) {
      AwesomeDialog(
              context: context,
              dialogType: DialogType.error,
              animType: AnimType.scale,
              showCloseIcon: true,
              title: "Error",
              desc: "Your password and confirm password do not match",
              btnCancelOnPress: () {},
              btnOkOnPress: () {})
          .show();
      return;
    }
    try {
      userCredential = await FirebaseAuth.instance
          .createUserWithEmailAndPassword(
              email: emailController.text, password: passwordController.text);
    } catch (e) {
      print("error: " + e.toString());
    }
    if (userCredential != null) {
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
            builder: (context) => HomePage(
                  title: '',
                )),
      );
    }
  }
}
