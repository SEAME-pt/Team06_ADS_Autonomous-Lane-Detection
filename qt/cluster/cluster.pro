QT += quick quickcontrols2

CONFIG += c++11

SOURCES += \
    main.cpp \
    clustercontroller.cpp

HEADERS += \
    clustercontroller.h

RESOURCES += qml.qrc

# Additional configuration for deployment
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
