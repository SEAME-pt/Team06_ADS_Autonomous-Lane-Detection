import pyopencl as cl

# Listar todas as plataformas OpenCL disponíveis
platforms = cl.get_platforms()

if not platforms:
    print("❌ Nenhuma plataforma OpenCL encontrada!")
else:
    print(f"✅ {len(platforms)} plataforma(s) OpenCL encontrada(s):")
    for i, platform in enumerate(platforms):
        print(f"\n🔹 Plataforma {i+1}: {platform.name}")
        print(f"   📌 Vendor: {platform.vendor}")
        print(f"   🔄 Versão: {platform.version}")

        # Listar dispositivos disponíveis nesta plataforma
        devices = platform.get_devices()
        if not devices:
            print("   ❌ Nenhum dispositivo disponível nesta plataforma.")
        else:
            print(f"   ✅ {len(devices)} dispositivo(s) encontrado(s):")
            for j, device in enumerate(devices):
                print(f"      🔸 Dispositivo {j+1}: {device.name}")
                print(f"         📦 Tipo: {cl.device_type.to_string(device.type)}")
                print(f"         🚀 Clock: {device.max_clock_frequency} MHz")
                print(f"         🔧 Unidades de Computação: {device.max_compute_units}")
                print(f"         🛠 Memória Global: {device.global_mem_size / 1024 / 1024:.2f} MB")


