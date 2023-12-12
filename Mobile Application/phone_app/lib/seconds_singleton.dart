class TimerSingleton {
  static late final TimerSingleton _singleton = TimerSingleton._internal();

  int seconds = 0;

  factory TimerSingleton() {
    return _singleton;
  }

  TimerSingleton._internal();
}
