import os
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

try:
    import laspy
except ImportError:
    raise ImportError("Для работы с .laz нужно установить laspy: pip install laspy")


def laz_to_stl(
    input_laz: str,
    output_stl: str,
    sample_rate: float = 0.05,
    scale_xy: float = 0.1,
    scale_z: float = 0.1,
    base_thickness: float = 50.0,
    base_height: float = 800.0,
    common_base: bool = False,
) -> str:
    """
    Конвертирует .laz в .stl:
      - читает облако точек из LAZ,
      - (опционально) предварительно отсэмпливает точки,
      - строит поверхность (Delaunay),
      - упрощает меш,
      - смещает к началу координат,
      - экструдирует дно,
      - масштабирует модель,
      - сохраняет STL-файл.

    Параметры:
        input_laz: путь к входному .laz файлу
        output_stl: путь к результирующему .stl файлу (директория будет создана)
        sample_rate: коэффициент decimation (0 < sample_rate < 1)
        scale_xy: масштаб по X/Y
        scale_z: масштаб по Z
        base_thickness: толщина базового экструдированного слоя (в единицах модели)
        base_height: если common_base=True, задаёт высоту общего основания
        common_base: если True — задаётся общая плоскость основания на высоте base_height

    Возвращает:
        output_stl — путь к сохранённому файлу
    """
    input_path = Path(input_laz)
    if not input_path.exists():
        raise FileNotFoundError(f"Входной файл не найден: {input_laz}")

    # Загрузка точек
    cloud = laspy.read(str(input_path))
    points = np.vstack((cloud.x, cloud.y, cloud.z)).T

    # Предварительный downsample (если нужно)
    if sample_rate < 1.0:
        skip = max(1, int(1.0 / sample_rate))
        points = points[::skip]

    # 1) Построение поверхности
    surf = pv.wrap(points).delaunay_2d(alpha=0.0)
    surf.compute_normals(inplace=True)

    # 2) Упрощение меша
    surf = surf.decimate(1.0 - sample_rate)

    # Гарантируем, что выходная директория существует
    output_path = Path(output_stl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Временно сохраняем промежуточный STL
    temp_stl = output_path.parent / (output_path.stem + "_tmp.stl")
    surf.save(str(temp_stl))

    # 3) Трансляция к началу координат
    mesh = trimesh.load(str(temp_stl))
    bbox_min, bbox_max = mesh.bounds
    # Z-смещение: либо к общей базе, либо к нижней грани меша
    if common_base:
        z_trans = -base_height
    else:
        z_trans = -bbox_min[2]
    translation = [-bbox_min[0], -bbox_min[1], z_trans]
    mesh.apply_translation(translation)

    # 4) Экструдирование дна
    size = bbox_max - bbox_min
    # создаём плоскость-ограничитель для extrude_trim
    plane = pv.Plane(
        center=(size[0] / 2, size[1] / 2, -base_thickness * (1 - scale_z)),
        direction=(0, 0, -1),
        i_size=size[0],
        j_size=size[1],
    )
    extruded = pv.wrap(mesh.vertices, mesh.faces.reshape((-1, 4))[:, 1:]) \
                  .extrude_trim((0, 0, -1.0), plane)

    # 5) Масштабирование
    final_mesh = extruded.scale([scale_xy, scale_xy, scale_z], inplace=False)

    # Сохраняем итоговый STL
    final_mesh.save(str(output_path))

    # Удаляем временный файл
    try:
        temp_stl.unlink()
    except Exception:
        pass

    return str(output_path)


if __name__ == "__main__":
    # Пример вызова (можно убрать в WebODM-плагине)
    import argparse

    parser = argparse.ArgumentParser(description="LAZ → STL конвертер")
    parser.add_argument("input_laz", help="Путь к входному .laz")
    parser.add_argument("output_stl", help="Куда сохранить .stl")
    parser.add_argument("--sample_rate", type=float, default=0.05)
    parser.add_argument("--scale_xy", type=float, default=0.1)
    parser.add_argument("--scale_z", type=float, default=0.1)
    parser.add_argument("--base_thickness", type=float, default=50.0)
    parser.add_argument("--base_height", type=float, default=800.0)
    parser.add_argument("--common_base", action="store_true")
    args = parser.parse_args()

    out = laz_to_stl(
        args.input_laz,
        args.output_stl,
        sample_rate=args.sample_rate,
        scale_xy=args.scale_xy,
        scale_z=args.scale_z,
        base_thickness=args.base_thickness,
        base_height=args.base_height,
        common_base=args.common_base,
    )
    print(f"✅ STL сохранён в: {out}")