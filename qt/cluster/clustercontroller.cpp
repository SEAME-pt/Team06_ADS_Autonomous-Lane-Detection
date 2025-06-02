
#include "clustercontroller.h"
#include <math.h>

ClusterController::ClusterController(QObject *parent)
    : QObject(parent)
{
    connect(&m_timer, &QTimer::timeout, this, &ClusterController::updateValues);
    connect(&m_tresources, &QTimer::timeout, this, &ClusterController::updateResources);

    updateSystemInfo();

    m_timer.start(50);  // Update every 50ms
    m_tresources.start(1000);


}

void ClusterController::updateResources()
{
    updateSystemInfo();

}

void ClusterController::updateValues()
{
    // Simulate changes - replace with real data
    m_speed = (m_speed + 0.1);
    if (m_speed > 15.0) m_speed = 0.0;
    
    m_battery = qMax(0, m_battery - 1);
    if (m_battery <= 0) m_battery = 100;
    
    m_direction = fmod(m_direction + 1.0, 360.0);
    
    emit speedChanged();

    emit batteryChanged();
    emit directionChanged();

}

