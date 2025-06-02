import QtQuick 2.15
import QtQuick.Controls 2.15
import QtGraphicalEffects 1.15

ApplicationWindow {
    visible: true
    width: 1280
    height: 480
    title: "Modern Automotive Cluster"
    color: "#1a1a1a"

    // Propriedades do velocímetro
    property color arcColor: currentSpeed > 9 ? "#ff4444" : "#00ff00"
    property real startAngle: -225
    property real spanAngle: 270
    property real currentAngle: (currentSpeed / maxSpeed) * spanAngle

    property real maxSpeedKMH: 15.0
    property real maxSpeedMPH: maxSpeedKMH * 0.621371
    property real currentSpeed: controller.isMetric ? controller.speed : controller.speedMPH
    property real maxSpeed: controller.isMetric ? maxSpeedKMH : maxSpeedMPH

    Rectangle {
        anchors.fill: parent
        color: "#1a1a1a"

        // Velocímetro

        // Fundo do velocímetro (arco cinza e números fixos)
        Canvas {
            id: backgroundArc
            anchors.fill: parent
            onPaint: {
                var ctx = getContext("2d");
                ctx.clearRect(0, 0, width, height);
                var centerX = width / 2;
                var centerY = height / 2;
                var radius = Math.min(width, height) / 2 - 40;


                ctx.beginPath();
                ctx.strokeStyle = "#333333";
                ctx.lineWidth = 32;
                ctx.lineCap = "round";
                var startRad = startAngle * Math.PI / 180;
                var endRad = (startAngle + spanAngle) * Math.PI / 180;
                ctx.arc(centerX, centerY, radius, startRad, endRad);
                ctx.stroke();


                var maxVal = controller.isMetric ? 15 : Math.ceil(15 * 0.621371);
                var step = controller.isMetric ? 1 : 1;

                for(var i = 0; i <= maxVal; i += step) {
                    var angle = startAngle + (i / maxVal) * spanAngle;
                    var angleRad = angle * Math.PI / 180;

                   // if (i % (controller.isMetric ? 1 : 2) === 0) {

                        ctx.save();
                        ctx.translate(
                            centerX + (radius - 45) * Math.cos(angleRad),
                            centerY + (radius - 35) * Math.sin(angleRad)
                        );
                        ctx.rotate(angleRad + Math.PI / 2);
                        ctx.fillStyle = "#88aa88";
                        ctx.font = "28px Arial";
                        ctx.textAlign = "center";
                        ctx.fillText(i.toString(), 0, 0);
                        ctx.restore();
                 //   }
                }
            }
        }

        // Arco de progresso da velocidade
        Canvas {
            id: progressArc
            anchors.fill: parent

            property real animatedAngle: currentAngle


            NumberAnimation on animatedAngle {
                id: angleAnimation
                from: progressArc.animatedAngle
                to: currentAngle
                duration: 300
                easing.type: Easing.OutCubic
            }

            onPaint: {
                var ctx = getContext("2d");
                ctx.clearRect(0, 0, width, height);
                var centerX = width / 2;
                var centerY = height / 2;
                var radius = Math.min(width, height) / 2 - 40;


                ctx.beginPath();
                ctx.strokeStyle = arcColor;
                ctx.lineWidth = 20;
                ctx.lineCap = "round";
                var startRad = startAngle * Math.PI / 180;
                var endRad = (startAngle + animatedAngle) * Math.PI / 180;
                ctx.arc(centerX, centerY, radius, startRad, endRad);
                ctx.stroke();
            }

            onAnimatedAngleChanged: requestPaint()
            Component.onCompleted: animatedAngle = currentAngle
        }


        Text {
            id: speedValue
            anchors.centerIn: parent
            text: Math.round(currentSpeed * 10) / 10
            color: arcColor
            font.pixelSize: 80
            font.family: "Arial"
            font.bold: true

            Behavior on color {
                ColorAnimation { duration: 200 }
            }
        }

        // Botão para alternar entre km/h e mph
        Rectangle {
            anchors.top: speedValue.bottom
            anchors.topMargin: 10
            anchors.horizontalCenter: parent.horizontalCenter
            width: 80
            height: 30
            color: "#333333"
            radius: 15

            MouseArea {
                anchors.fill: parent
                onClicked: controller.setIsMetric(!controller.isMetric)
            }

            Text {
                anchors.centerIn: parent
                text: controller.speedUnit
                color: arcColor
                font.pixelSize: 16
                font.bold: true

                Behavior on color {
                    ColorAnimation { duration: 200 }
                }
            }
        }

        // Atualiza o velocímetro quando a velocidade muda
        Connections {
            target: controller
            onSpeedChanged: {
                angleAnimation.restart()
                progressArc.requestPaint()
            }
        }




        // Battery
        Rectangle {
            id: batteryIndicator
            width: parent.width * 0.040
            height: parent.height * 0.1
            anchors.right: parent.right
            anchors.rightMargin: 80
            anchors.top: parent.top
            anchors.topMargin: 80
            color: "transparent"
            border.color: "#333333"
            border.width: 2
            radius: 10



            Text {
                    id: batteryIcon
                    x: 5
                    y: parent.height + height - 5
                    font.family: faFont
                    font.pixelSize: 50
                    text: {
                        if (controller.battery > 75) return "\uf240"; // Full Battery
                        if (controller.battery > 50) return "\uf241"; // Half Battery
                        if (controller.battery > 25) return "\uf242"; // Low Battery
                        return "\uf243"; // Empty Battery
                    }
                    color: controller.battery < 20 ? "#ff0000" : "#00ff00"
                    anchors.centerIn: parent
                }

            Text {
                anchors.horizontalCenter: parent.horizontalCenter
                y: parent.height + 10
                text: controller.battery + "%"
                color: "#00ff00"
                font.pixelSize: 24
            }



        }

        // Direction
        /*
        Item {
            id: compass
            width: parent.width * 0.16
            height: width
            x: (width / 2)*0.1
            y:(parent.height / 2) - (height / 2)  * 0.1


            Rectangle {
                id: compassBackground
                anchors.fill: parent
                radius: width / 2
                color: "transparent"
                border.color: "#333333"
                border.width: 2

                // Círculo interno
                Rectangle {
                    anchors.centerIn: parent
                    width: parent.width * 0.9
                    height: width
                    radius: width / 2
                    color: "transparent"
                    border.color: "#222222"
                    border.width: 1
                }

                // Pontos cardeais
                Repeater {
                    model: ["N", "NE", "L", "SE", "S", "SO", "O", "NO"]
                    Text {
                        x: compassBackground.width/2 - width/2 + (compassBackground.width/2 - 30) * Math.cos((index * 45 - 90) * Math.PI/180)
                        y: compassBackground.height/2 - height/2 + (compassBackground.height/2 - 30) * Math.sin((index * 45 - 90) * Math.PI/180)
                        text: modelData
                        color: modelData === "N" ? "#00ff00" : "#888888"
                        font.pixelSize: modelData === "N" ? 26 : 20
                        font.bold: modelData === "N"
                    }
                }

                // Marcadores de graus
                Canvas {
                    anchors.fill: parent
                    onPaint: {
                        var ctx = getContext("2d");
                        var centerX = width / 2;
                        var centerY = height / 2;
                        var radius = width / 2 - 5;

                        for(var i = 0; i < 360; i += 15) {
                            var angle = (i - 90) * Math.PI / 180;
                            var length = (i % 45 === 0) ? 15 : 8;

                            var startX = centerX + (radius - length) * Math.cos(angle);
                            var startY = centerY + (radius - length) * Math.sin(angle);
                            var endX = centerX + radius * Math.cos(angle);
                            var endY = centerY + radius * Math.sin(angle);

                            ctx.beginPath();
                            ctx.strokeStyle = "#444444";
                            ctx.lineWidth = (i % 45 === 0) ? 2 : 1;
                            ctx.moveTo(startX, startY);
                            ctx.lineTo(endX, endY);
                            ctx.stroke();
                        }
                    }
                }

                // Ponteiro de direção
                Item {
                    id: directionPointer
                    anchors.centerIn: parent
                    width: parent.width
                    height: parent.height
                    rotation: controller.direction

                    Behavior on rotation {
                        NumberAnimation {
                            duration: 100
                            easing.type: Easing.OutCubic
                        }
                    }

                    Rectangle {
                        id: pointer
                        width: 4
                        height: parent.height * 0.4
                        color: "#0088ff"
                        radius: 2
                        antialiasing: true
                        x: parent.width/2 - width/2
                        y: parent.height/2 - height

                        // Ponta do ponteiro
                        Rectangle {
                            width: 12
                            height: 12
                            radius: 6
                            color: pointer.color
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.bottom: parent.top
                            anchors.bottomMargin: -6
                        }

                        // Base do ponteiro
                        Rectangle {
                            width: 8
                            height: 8
                            radius: 4
                            color: pointer.color
                            anchors.horizontalCenter: parent.horizontalCenter
                            anchors.top: parent.bottom
                            anchors.topMargin: -4
                        }
                    }
                }

                // Valor numérico da direção
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 8
                    text: controller.direction + "°"
                    color: "#00ff00"
                    font.pixelSize: 16
                }


            }
        }
*/

        Item {
            id: systemMonitor
            width: parent.width * 0.2
            height: parent.height * 0.1
            anchors.right: parent.right
            anchors.rightMargin: 20
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 80

            // CPU Usage
            Rectangle {
                width: parent.width
                height: 40
                color: "#222222"
                radius: 10

                Text {
                    anchors.left: parent.left
                    anchors.leftMargin: 10
                    anchors.verticalCenter: parent.verticalCenter
                    text: "CPU: "+ controller.cpuUsage
                    color: "#ffffff"
                    font.pixelSize: 18
                }

                Rectangle {
                    id: cpuBar
                    width: parent.width * (controller.cpuUsage / 100)
                    height: parent.height
                    color: controller.cpuUsage > 80 ? "#ff4444" : "#00ff00" // Vermelho se > 80%
                    radius: 10

                    Behavior on width {
                        NumberAnimation { duration: 200 }
                    }
                }
            }

            // RAM Usage
            Rectangle {
                width: parent.width
                height: 40
                color: "#222222"
                radius: 10
                anchors.top: parent.bottom
                anchors.topMargin: 20

                Text {
                    anchors.left: parent.left
                    anchors.leftMargin: 10
                    anchors.verticalCenter: parent.verticalCenter
                    text: "RAM: "+ controller.ramFree +"/"+ controller.ramTotal
                    color: "#ffffff"
                    font.pixelSize: 18
                }

                Rectangle {
                    id: ramBar
                    width: parent.width * (controller.ramUsage / 100)
                    height: parent.height
                    color: controller.ramUsage > 80 ? "#ff4444" : "#00ff00" // Vermelho se > 80%
                    radius: 10

                    Behavior on width {
                        NumberAnimation { duration: 200 }
                    }
                }
            }
        }




    }

}
