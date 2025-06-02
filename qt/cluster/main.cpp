

#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QFontDatabase>
#include "clustercontroller.h"

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    
    ClusterController controller;
    QQmlApplicationEngine engine;
    
    int fontId = QFontDatabase::addApplicationFont(":/resources/fonts/fa-solid-900.ttf");
    QString fontFamily = QFontDatabase::applicationFontFamilies(fontId).at(0);

    engine.rootContext()->setContextProperty("faFont", fontFamily);


    engine.rootContext()->setContextProperty("controller", &controller);
    engine.load(QUrl(QStringLiteral("qrc:/qml/main.qml")));

    
    return app.exec();
}

