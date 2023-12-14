// Real-time sampling and ECG output

const int heartPin = A1;
void setup() {
  Serial.begin(115200);
}
void loop() {
int heartValue = analogRead(heartPin);
Serial.println(heartValue);
delay(5);
}
