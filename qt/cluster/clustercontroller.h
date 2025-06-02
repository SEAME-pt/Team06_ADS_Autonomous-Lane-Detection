// clustercontroller.h
#ifndef CLUSTERCONTROLLER_H
#define CLUSTERCONTROLLER_H

#include <QObject>
#include <QTimer>
#include <QProcess>


class ClusterController : public QObject
{
    Q_OBJECT
    Q_PROPERTY(double speed READ speed NOTIFY speedChanged)
    Q_PROPERTY(double speedMPH READ speedMPH NOTIFY speedChanged)
    Q_PROPERTY(int battery READ battery NOTIFY batteryChanged)
    Q_PROPERTY(double direction READ direction NOTIFY directionChanged)
    Q_PROPERTY(bool isMetric READ isMetric WRITE setIsMetric NOTIFY unitChanged)
    Q_PROPERTY(QString speedUnit READ speedUnit NOTIFY unitChanged)
    Q_PROPERTY(QString cpuUsage READ cpuUsage NOTIFY systemInfoUpdated)
    Q_PROPERTY(QString ramTotal READ ramTotal NOTIFY systemInfoUpdated)
    Q_PROPERTY(QString ramFree READ ramFree NOTIFY systemInfoUpdated)

public:
    explicit ClusterController(QObject *parent = nullptr);

    double speed() const { return m_speed; }
    double speedMPH() const { return m_speed * 0.621371; }
    int battery() const { return m_battery; }
    double direction() const { return m_direction; }
    bool isMetric() const { return m_isMetric; }
    QString speedUnit() const { return m_isMetric ? "km/h" : "mph"; }



    QString cpuUsage() const { return m_cpuUsage; }
    QString ramTotal() const { return m_ramTotal; }
    QString ramFree() const { return m_ramFree; }

public slots:
    void setIsMetric(bool metric)
    {
        if (m_isMetric != metric)
        {
            m_isMetric = metric;
            emit unitChanged();
        }
    }

    void updateSystemInfo()
    {
        QProcess process;

        // CPU Usage
        process.start("top", QStringList() << "-bn1");
        process.waitForFinished();
        QStringList cpuLines = QString(process.readAllStandardOutput()).split("\n");
        for (QString line : cpuLines) {
            if (line.contains("Cpu(s)")) {
                m_cpuUsage = line.split(",")[0].split(":")[1].trimmed() + "%";
                break;
            }
        }

        // RAM Usage
        process.start("free", QStringList() << "-m");
        process.waitForFinished();
        QStringList ramLines = QString(process.readAllStandardOutput()).split("\n");
        if (ramLines.size() > 1) {
            QStringList ramValues = ramLines[1].split(QRegExp("\\s+"));
            if (ramValues.size() > 2) {
                m_ramTotal = ramValues[1] + " MB";
                m_ramFree = ramValues[3] + " MB";
            }
        }
          emit systemInfoUpdated();
    }

signals:
    void speedChanged();
    void batteryChanged();
    void directionChanged();
    void unitChanged();
    void systemInfoUpdated();

private slots:
    void updateValues();
    void updateResources();

private:
    QTimer m_timer;
    QTimer m_tresources;

    double m_speed{0.0};
    int m_battery{100};
    double m_direction{0.0};
    bool m_isMetric{true};
    QString m_cpuUsage;
    QString m_ramTotal;
    QString m_ramFree;
};

#endif
