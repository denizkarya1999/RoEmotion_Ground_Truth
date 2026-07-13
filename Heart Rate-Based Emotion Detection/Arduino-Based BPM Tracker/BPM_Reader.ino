// Pulse sensor wiring:
// S -> A0
// + -> 5V
// - -> GND

const int pulsePin = A0;

void setup() {
  Serial.begin(115200);
}

void loop() {
  int pulseValue = analogRead(pulsePin);

  // Send timestamp and sensor reading to Python.
  Serial.print(millis());
  Serial.print(",");
  Serial.println(pulseValue);

  delay(10);  // Approximately 100 samples per second
}
