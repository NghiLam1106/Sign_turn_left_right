#include <Servo.h>
Servo servoLeft ;      // Define left servo
Servo servoRight;      // Define right servo

int trig = 2;          //Trig pin
int echo = 3;          //Echo pin
int servo_rightpin=9;
int servo_leftpin=10;

void setup() {
  Serial.begin(9600);
  pinMode(trig,OUTPUT); //Trigger is an output pin
  pinMode(echo,INPUT);  // Echo is an input pin
  servoRight.attach(servo_rightpin);  
  servoLeft.attach(servo_leftpin);    
  stopRobot();
  delay(2000);
}

void loop() {
  float Length, distance;

  digitalWrite(trig, LOW);     //초기화
  delay(2);
  digitalWrite(trig, HIGH);    // trigger 신호 발생 (10us)
  delay(10);
  digitalWrite(trig, LOW);
  
  Length = pulseIn(echo, HIGH);  // Echo 신호 입력, pulseIn값 저장
  distance = ((float)(340 * Length) / 10000) / 2;    // 거리 측정 및 계산
  Serial.print(distance);   Serial.println(" cm");    // Serial 출력
  delay(500);  // 0.5sec마다 출력

  if(distance < 45.00){
    stopRobot(); 
    delay(500);
    reverse();   
    delay(1000);
    turnRight(); 
    delay(1000);
    forward();
  }
  else{
    forward();
  }
}

void forward() {
  servoLeft.write(180);
  servoRight.write(0);
}

void reverse() {
  servoLeft.write(0);
  servoRight.write(180);
}

void turnRight() {
  servoLeft.write(180);
  servoRight.write(90);
}

void turnLeft() {
  servoLeft.write(90);
  servoRight.write(0);
}

void stopRobot() {
  servoLeft.write(90);
  servoRight.write(90);
}
