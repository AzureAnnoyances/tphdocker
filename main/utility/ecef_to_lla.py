import laspy
def is_Geodetic(las_file_path):
    """Comprehensive check for georeferencing"""
    with laspy.open(las_file_path) as las:
        header = las.header
        has_wkt = header.global_encoding.wkt
        rtn_string = f"\n=== Georeferencing Check ===\nFile : [{las_file_path}]\n"
        if has_wkt:
            rtn_string += "Found   [wkt], \n"
            
            crs = las.header.parse_crs()
            if crs:
                print(crs.to_epsg())
                rtn_string += "Found   [CRS Geodetic] \n"
            else:
                rtn_string += "Missing [CRS Geodetic], \n"
            
            geo_reference_record_ids = [34735, 34736, 34737]
            vlr_ids = [vlr.record_id for vlr in las.header.vlrs]
            vlr_geo_records = [vlr_id in geo_reference_record_ids for vlr_id in vlr_ids]
            
            if any(vlr_geo_records) > 0:
                rtn_string += f"Found   [vlr_geo_records] : [{vlr_geo_records}]\n"
                return True, rtn_string
            else:
                rtn_string += f"Missing [vlr_geo_records]\n"
                return False, rtn_string
        else:
            rtn_string = "Missing [Geo Record], \nMissing [CRS Geodetic], \nMissing [vlr_geo_records]"
            
            return False, rtn_string