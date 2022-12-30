### QGIS
from qgis.PyQt.QtCore import QCoreApplication, QSettings, QVariant
from qgis.PyQt.QtWidgets import (
    QWidget,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QLineEdit,
    QLabel,
    QCheckBox,
)
from qgis.PyQt.QtGui import QColor, QDoubleValidator
from qgis.core import (
    QgsContrastEnhancement,
    QgsColorRampShader,
    QgsFeature,
    QgsFeatureRequest,
    QgsFeatureSink,
    QgsField,
    QgsGeometry,
    QgsGraduatedSymbolRenderer,
    QgsMessageLog,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsRendererRange,
    QgsSimpleMarkerSymbolLayer,
    QgsSingleBandPseudoColorRenderer,
    QgsSymbol,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.gui import *
from qgis.utils import *

# Python libraries
from osgeo import gdal
import time
import numpy as np
import os
import struct
import itertools
import math
from datetime import date
from scipy import interpolate, signal, stats, ndimage
from scipy.spatial import Delaunay  # ConvexHull
from scipy.special import gamma, erf
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
import gc

# import tracemalloc

# GUI class in QT
class Buttons(QWidget):
    def processar_simrad(self):
        btn_num = 0
        pixel_size = 1.0  # 1.17
        run_checksum = False
        # Get files from user
        fs, __ = QFileDialog.getOpenFileNames(
            QFileDialog(),
            self.btn_names[btn_num],
            self.get_cur_path(),
            self.data_types[btn_num],
        )
        self.update_textbox("[START] " + self.btn_names[btn_num])
        # For each indicated file
        for f in fs:
            # tracemalloc.start(25)
            # Split file and folder names
            folder_name, file_name = os.path.split(f)
            self.set_cur_path(folder_name)
            self.update_textbox(file_name + "\n\tReading file", "TIME_START")
            pos_dgs, sid_dgs, xyz_dgs, rra_dgs, run_dgs = read_datagrams(
                f, run_checksum
            )
            gc.collect()
            self.update_textbox(
                file_name + "\n\tArranging datagrams to create waterfall", "TIME_DIFF"
            )
            # Get pings for SID, XYZ and RRA datagrams
            png_sid = np.asarray([dg["ping"] for dg in sid_dgs])
            png_xyz = np.asarray([dg["ping"] for dg in xyz_dgs])
            png_rra = np.asarray([dg["ping"] for dg in rra_dgs])
            if png_sid.size == 0:
                self.plugin_warning(
                    file_name
                    + ': file does not contain datagrams for "seabed image data 89"'
                )
                self.plugin_warning(".")
            if png_rra.size == 0:
                self.plugin_warning(
                    file_name
                    + ': file does not contain datagrams for "raw range and angle 78"'
                )
                self.plugin_warning(".")
            # Find matching pings for SID, XYZ and RRA datagrams
            png_unq = np.intersect1d(np.intersect1d(png_sid, png_xyz), png_rra)
            sid_dgs = [
                sid_dgs[k]
                for k in np.intersect1d(png_sid, png_unq, return_indices=True)[1]
            ]
            rra_dgs = [
                rra_dgs[k]
                for k in np.intersect1d(png_rra, png_unq, return_indices=True)[1]
            ]
            xyz_dgs = [
                xyz_dgs[k]
                for k in np.intersect1d(png_xyz, png_unq, return_indices=True)[1]
            ]
            # Sort datagrams by time
            ts = get_datenum(sid_dgs)
            if ts != sorted(ts):
                tsi = np.argsort(ts)
                sid_dgs = [sid_dgs[i] for i in tsi]
                rra_dgs = [rra_dgs[i] for i in tsi]
                xyz_dgs = [xyz_dgs[i] for i in tsi]
                del tsi
            del ts
            gc.collect()
            self.update_textbox(file_name + "\n\tCreating waterfall", "TIME_DIFF")
            (
                wtfall_img,
                angles_img,
                reflec_img,
                bad_val,
                centre_freq,
                sound_speed,
            ) = create_waterfall(sid_dgs, xyz_dgs, rra_dgs, pixel_size, run_dgs)
            # Save calibration data for current file
            fp = open("%s_calib.txt" % f, "w")
            fp.write("EM_model_number: %d\n" % pos_dgs[0]["em_model"])
            fp.write("Center_frequency: %f\n" % centre_freq)
            fp.write("Sound_speed: %f\n" % sound_speed)
            fp.close()
            gc.collect()
            self.update_textbox(
                file_name + "\n\tGetting GCPs for waterfall", "TIME_DIFF"
            )
            gcpList, cur_epsg_code = get_ground_control_points(
                pos_dgs, xyz_dgs, wtfall_img, pixel_size, bad_val
            )
            gc.collect()
            self.update_textbox(
                file_name + "\n\tAdjusting values for waterfall", "TIME_DIFF"
            )
            angles_img = image_normalization(
                angles_img,
                bad_val,
                self.new_bad_val,
                self.angl_corr[0],
                self.angl_corr[1],
                self.angl_corr[2],
                self.angl_corr[3],
            )
            wtfall_img = image_normalization(
                wtfall_img,
                bad_val,
                self.new_bad_val,
                self.back_corr[0],
                self.back_corr[1],
                self.back_corr[2],
                self.back_corr[3],
            )
            reflec_img = image_normalization(
                reflec_img,
                bad_val,
                self.new_bad_val,
                self.back_corr[0],
                self.back_corr[1],
                self.back_corr[2],
                self.back_corr[3],
            )
            gc.collect()
            self.update_textbox(
                file_name + "\n\tApplying median filter to waterfall", "TIME_DIFF"
            )
            wtf_nok = wtfall_img == self.new_bad_val
            wtfall_img = np.clip(
                signal.medfilt2d(wtfall_img, (3, 5)),
                self.back_corr[4],
                self.back_corr[5],
            )
            wtfall_img[wtf_nok] = self.new_bad_val
            gc.collect()
            self.update_textbox(
                file_name + "\n\tApplying notch filter to waterfall", "TIME_DIFF"
            )
            wtfall_img = np.clip(
                notch_filtering(wtfall_img, 0), self.back_corr[4], self.back_corr[5]
            )
            wtfall_img[wtf_nok] = self.new_bad_val
            gc.collect()
            self.update_textbox(
                file_name + "\n\tSaving backscatter waterfall", "TIME_DIFF"
            )
            save_numpy_array(
                np.round(wtfall_img).astype(self.img_output_type), "%s_back_wtf.tif" % f
            )
            gc.collect()
            self.update_textbox(file_name + "\n\tSaving angles waterfall", "TIME_DIFF")
            save_numpy_array(
                angles_img.astype(self.img_output_type), "%s_angl_wtf.tif" % f
            )
            gc.collect()
            self.update_textbox(
                file_name + "\n\tSaving reflectivity waterfall", "TIME_DIFF"
            )
            save_numpy_array(
                reflec_img.astype(self.img_output_type), "%s_refl_wtf.tif" % f
            )
            gc.collect()
            file_types = ["back", "angl", "refl"]
            for ft in file_types:  # range(len(file_types)):
                wtf_tiff = "%s_%s_wtf.tif" % (f, ft)
                geo_out = "%s_%s_geo%s" % (f, ft, self.output_ext)
                self.update_textbox(
                    file_name + "\n\tGeoreferencing %s waterfall" % ft, "TIME_DIFF"
                )
                translate_warp_tif(
                    wtf_tiff, gcpList, cur_epsg_code, geo_out, True, self.save_wtf
                )
                gc.collect()
                if self.output_ext == ".jp2":
                    self.update_textbox(
                        file_name
                        + "\n\tConverting georeferenced %s waterfall from TIFF to JP2"
                        % ft,
                        "TIME_DIFF",
                    )
                    tiff_to_jp2(wtf_tiff, gcpList, cur_epsg_code)
                    gc.collect()
            self.update_textbox(
                file_name + "\n\tLoading georeferenced waterfalls", "TIME_DIFF"
            )
            load_image_QGIS(
                "%s_back_geo%s" % (f, self.output_ext),
                "Backscatter_" + file_name,
                self.back_corr[4],
                self.back_corr[5],
            )
            # print(np.diff(np.asarray(t0)))
            # snapshot = tracemalloc.take_snapshot()
            # top_stats = snapshot.statistics('traceback')
            # stat = top_stats[0]
            # print("%s memory blocks: %.1f MB" % (stat.count, stat.size / (2**20)))
        self.update_textbox("[END] " + self.btn_names[btn_num], "TIME_DIFF_LAST")

    def carregar_simrad(self):
        btn_num = 1
        fs, __ = QFileDialog.getOpenFileNames(
            QFileDialog(),
            self.btn_names[btn_num],
            self.get_cur_path(),
            self.data_types[btn_num],
        )
        self.update_textbox("[START] " + self.btn_names[btn_num])
        for f in fs:
            folder_name, file_name = os.path.split(f)
            self.set_cur_path(folder_name)
            self.update_textbox("Reading " + file_name, "TIME_START")
            load_image_QGIS(
                "%s_back_geo%s" % (f, self.output_ext),
                "Backscatter_" + file_name,
                self.back_corr[4],
                self.back_corr[5],
            )
            # load_image_QGIS("%s_angl_geo.tif" % f, "Angulos_" + file_name, -90+180, 90+180)
            # load_image_QGIS("%s_refl_geo.tif" % f, "Reflexoes_" + file_name, 5000, 10000)
        self.update_textbox("[END] " + self.btn_names[btn_num], "TIME_DIFF_LAST")

    def processar_shapefile(self):
        btn_num = 2
        registry = QgsProject.instance()
        # registry = QgsMapLayerRegistry.instance()
        shape_files, __ = QFileDialog.getOpenFileNames(
            QFileDialog(),
            self.btn_names[btn_num],
            self.get_cur_path(),
            self.data_types[btn_num],
        )
        self.update_textbox("[START] " + self.btn_names[btn_num])
        for cur_shape in shape_files:
            layers = registry.mapLayers()
            folder_name, file_name = os.path.split(cur_shape)
            self.set_cur_path(folder_name)
            self.update_textbox(
                file_name
                + "\n\tSearching for intersections with georeferenced waterfalls",
                "TIME_START",
            )
            # ang_georef_cut = []
            # ref_georef_cut = []
            hist_angls = []
            hist_refls = []
            hist_refls_without_angs = np.array(0)
            hist_cnts = []
            freq = 0
            nu_water = 0
            freq_N = 0
            for l in layers.keys():
                cur_raster = layers[l].source()
                if (
                    cur_raster.find(self.output_ext) > -1
                    and cur_raster.find(".all") > -1
                ):
                    cur_raster = cur_raster[0:-13]  # Remove '_back_geo.tif'
                    raster_folder_name, raster_file_name = os.path.split(cur_raster)
                    ang_georef_tif = "%s_angl_geo%s" % (cur_raster, self.output_ext)
                    ref_georef_tif = "%s_refl_geo%s" % (cur_raster, self.output_ext)
                    ang_georef_cut = "%s_%s_angl_geo_cut%s" % (
                        cur_shape,
                        raster_file_name,
                        self.output_ext,
                    )
                    ref_georef_cut = "%s_%s_refl_geo_cut%s" % (
                        cur_shape,
                        raster_file_name,
                        self.output_ext,
                    )
                    # Procurar intersecoes do shapefile com a imagem de angulos
                    get_shape_cut(ang_georef_tif, cur_shape, ang_georef_cut)
                    ds = gdal.Open(ang_georef_cut)

                    # Se houve alguma intersecao
                    # if ds!=None and ds.GetRasterBand(1).GetStatistics(0,1)!=[0.0,0.0,0.0,0.0]:
                    if (
                        ds != None
                        and len(np.unique(ds.GetRasterBand(1).ReadRaster())) != 1
                    ):
                        self.update_textbox(
                            "Found intersection with file:\n\t" + ang_georef_tif,
                            "TIME_DIFF",
                        )
                        # Obter dados de calibracao
                        with open("%s_calib.txt" % cur_raster, "r") as fp:
                            calib_txt, model_num = fp.readline().split()
                            calib_txt, cur_centre_freq = fp.readline().split()
                            calib_txt, cur_sound_speed = fp.readline().split()
                            freq += float(cur_centre_freq)
                            nu_water += float(cur_sound_speed)
                            freq_N += 1
                        # Obter intersecoes com imagem de reflectancia
                        get_shape_cut(ref_georef_tif, cur_shape, ref_georef_cut)
                        # Calcular histograma de reflectancia em funcao do angulo
                        angl_inv_corr = (
                            -self.angl_corr[3] / self.angl_corr[2],
                            1 / self.angl_corr[2],
                        )
                        back_inv_corr = (
                            -self.back_corr[3] / self.back_corr[2],
                            1 / self.back_corr[2],
                        )
                        hist_angls, cur_refls, cur_cnts = calc_hist_by_angle(
                            ang_georef_cut,
                            ref_georef_cut,
                            angl_inv_corr,
                            back_inv_corr,
                            self.back_corr[0],
                            True,
                        )
                        hist_refls_without_angs = np.append(
                            hist_refls_without_angs, read_img(ref_georef_cut)
                        )
                        if len(hist_refls) == 0:
                            hist_refls = cur_refls
                            hist_cnts = cur_cnts
                        else:
                            hist_refls += cur_refls
                            hist_cnts += cur_cnts
                    else:
                        self.update_textbox(
                            "No intersections found with file:\n\t" + ang_georef_tif,
                            "TIME_DIFF",
                        )
                    # Apagar arquivos de intersecao
                    del ds
                    force_remove(ang_georef_cut)
                    force_remove(ref_georef_cut)
                    force_remove(("%s.aux.xml" % ang_georef_cut))
            if len(hist_angls) > 0:
                cur_vals1 = hist_cnts > 5
                hist_refls[cur_vals1] /= hist_cnts[cur_vals1]
                cur_vals2 = np.logical_not(cur_vals1)
                hist_refls[cur_vals2] = np.interp(
                    hist_angls[cur_vals2], hist_angls[cur_vals1], hist_refls[cur_vals1]
                )
                # hist_refls /= hist_cnts

                valid_angls = np.absolute(hist_angls) <= 60
                hist_refls = hist_refls[valid_angls]
                hist_angls = hist_angls[valid_angls]

                ##############################
                # LER NO ARQUIVO *_calib.txt #
                ##############################
                freq /= freq_N
                nu_water /= freq_N
                # freq = 30000 # frequencia dos beams
                # nu_water = 1460 # velocidade do som na aguas
                ##############################
                # LER NO ARQUIVO *_calib.txt #
                ##############################

                hist_refls_without_angs = (
                    hist_refls_without_angs[hist_refls_without_angs != self.new_bad_val]
                    * back_inv_corr[1]
                    + back_inv_corr[0]
                )
                hist_refls_without_angs = 10 ** (hist_refls_without_angs / 20.0)
                n_bins = 50
                Hin = np.histogram(hist_refls_without_angs, n_bins, range=(0.0, 1.0))
                rnw = RedeNeuralWeibull6(Hin[0])

                self.update_textbox(file_name + "\n\tInverting the model", "TIME_DIFF")
                inverted_model = Invert_Model(hist_angls, hist_refls, freq, nu_water)
                # inverted_model = Invert_Model(hist_angls, hist_refls, hist_cnts, freq, nu_water)
                inverted_model_layer = QgsVectorLayer(cur_shape, file_name, "ogr")
                inverted_model_layer.startEditing()
                inverted_model_layer.setOpacity(0.5)

                plt.figure()
                ax = plt.subplot(111)
                plt.plot(hist_angls, hist_refls, "-b", label="Real angular response")
                plt.plot(
                    inverted_model.angls, inverted_model.S_total, "-r", label="Model"
                )
                # plt.plot(inverted_model.m.deg, inverted_model.m.Sb_total, '-r', label='Modelo')
                axes = plt.gca()
                axes.set_ylim([-50, 0])
                plt.grid(True)
                plt.xlabel("Angles (degrees)")
                plt.ylabel("Average backscatter (dB)")
                # plt.title('%s\nImpedance=%f\nPhi=%f' %
                #     (file_name, inverted_model.impedance, inverted_model.phi_avo))
                plt.show()
                cur_shape_angle_response = "%s_angle_response.png" % cur_shape
                plt.savefig(cur_shape_angle_response)
                plt.close()

                # Remove shapefile if it is loaded
                for layer in QgsProject.instance().mapLayers().values():
                    if cur_shape.replace("\\", "/") == layer.source().replace(
                        "\\", "/"
                    ):
                        QgsProject.instance().removeMapLayer(layer.id())

                # Add it again, but now we can control it
                QgsProject.instance().addMapLayers([inverted_model_layer])
                attr_names = ["Impedance", "Vol", "Rugo", "Weibull_a", "Weibull_b"]
                attr_disp_names = [
                    "Impedance",
                    "Volume",
                    "Rugosity",
                    "Weibull_a",
                    "Weibull_b",
                ]
                attr_vals = [
                    float(inverted_model.impedance),
                    float(inverted_model.sigma2),
                    float(inverted_model.ch0),
                    float(rnw.a),
                    float(rnw.b),
                ]
                lyr_data_prov = inverted_model_layer.dataProvider()
                cur_fields = [field.name() for field in lyr_data_prov.fields()]
                for cur_name in attr_names:
                    if not any(cur_name in s for s in cur_fields):
                        lyr_data_prov.addAttributes(
                            [QgsField(cur_name, QVariant.Double)]
                        )
                inverted_model_layer.updateFields()

                inverted_model_layer.startEditing()
                for feat in lyr_data_prov.getFeatures():
                    for n in range(len(attr_names)):
                        feat.setAttribute(attr_names[n], attr_vals[n])
                    inverted_model_layer.updateFeature(feat)
                inverted_model_layer.commitChanges()

                cur_map_tip = '<div style="height:400;font-size:18px">\n'
                cur_map_tip += '\t<div style="height:110">\n\t\t<ul>\n'
                # cur_map_tip = '<div style="height:600;font-size:18px">\n'
                # cur_map_tip += '\t<div style="height:150">\n\t\t<ul>\n'
                for n in range(len(attr_names)):
                    if len(attr_disp_names[n]):
                        cur_map_tip += '\t\t\t<li>%s = [%% "%s" %%]</li>\n' % (
                            attr_disp_names[n],
                            attr_names[n],
                        )
                cur_map_tip += "\t\t</ul>\n\t</div>\n"
                cur_map_tip += '\t<div style="height:250">\n'
                cur_map_tip += '\t\t<img src="file:///%s" ' % cur_shape_angle_response
                cur_map_tip += 'style="max-height:250px" />\n'
                cur_map_tip += "\t</div>\n"
                cur_map_tip += "</div>"
                print(cur_map_tip)
                inverted_model_layer.startEditing()
                inverted_model_layer.setMapTipTemplate(cur_map_tip)
                inverted_model_layer.commitChanges()

                QgsProject.instance().addMapLayer(inverted_model_layer)
        self.update_textbox("[END] " + self.btn_names[btn_num], "TIME_DIFF_LAST")

    def carregar_shapefile(self):
        btn_num = 3
        registry = QgsProject.instance()
        # registry = QgsMapLayerRegistry.instance()
        layers = registry.mapLayers()
        shape_files, __ = QFileDialog.getOpenFileNames(
            QFileDialog(),
            self.btn_names[btn_num],
            self.get_cur_path(),
            self.data_types[btn_num],
        )
        self.update_textbox("[START] " + self.btn_names[btn_num])
        for cur_shape in shape_files:
            folder_name, file_name = os.path.split(cur_shape)
            self.set_cur_path(folder_name)
            inverted_model_layer = QgsVectorLayer(cur_shape, file_name, "ogr")
            inverted_model_layer.startEditing()
            inverted_model_layer.setOpacity(0.5)
            # Remove shapefile if it is loaded
            for layer in QgsProject.instance().mapLayers().values():
                if cur_shape.replace("\\", "/") == layer.source().replace("\\", "/"):
                    QgsProject.instance().removeMapLayer(layer.id())
            # Add it again, but now we can control it
            QgsProject.instance().addMapLayers([inverted_model_layer])
            # impedance = Impedance
            # sigma2 = volume
            # ch0 = rugosity
            attr_names = ["Impedance", "Vol", "Rugo", "Weibull_a", "Weibull_b"]
            attr_disp_names = [
                "Impedance",
                "Volume",
                "Rugosity",
                "Weibull_a",
                "Weibull_b",
            ]
            # attr_names = ['Impedance',
            #     'TamGrao',
            #     'VelSom',
            #     'Densidade',
            #     'Atenu',
            #     'Rugo',
            #     'Vol']
            # attr_disp_names = ['Impedance',
            #     'Tamanho do grão',
            #     'Velocidade do som',
            #     'Densidade',
            #     'Atenuação',
            #     'Rugosity',
            #     'Volume']
            lyr_data_prov = inverted_model_layer.dataProvider()
            cur_fields = [field.name() for field in lyr_data_prov.fields()]
            cur_map_tip = ""
            for cur_name in attr_names:
                if not any(cur_name in s for s in cur_fields):
                    cur_map_tip = "<div>Shapefile não processado!</div>"
                    break
            if len(cur_map_tip) == 0:
                cur_shape_angle_response = "%s_angle_response.png" % cur_shape
                cur_map_tip = '<div style="height:400;font-size:18px">\n'
                cur_map_tip += '\t<div style="height:110">\n\t\t<ul>\n'
                # cur_map_tip = '<div style="height:600;font-size:18px">\n'
                # cur_map_tip += '\t<div style="height:150">\n\t\t<ul>\n'
                for n in range(len(attr_names)):
                    if len(attr_disp_names[n]):
                        cur_map_tip += '\t\t\t<li>%s = [%% "%s" %%]</li>\n' % (
                            attr_disp_names[n],
                            attr_names[n],
                        )
                cur_map_tip += "\t\t</ul>\n\t</div>\n"
                cur_map_tip += (
                    '\t<div style="height:250">\n\t\t<img src="file:///%s"'
                    ' style="max-height:250px" />\n\t</div>\n'
                    % cur_shape_angle_response
                )
                cur_map_tip += "</div>"
            inverted_model_layer.setMapTipTemplate(cur_map_tip)
            inverted_model_layer.updateFields()
            inverted_model_layer.commitChanges()
            # inverted_model_layer.rollBack()
            QgsProject.instance().addMapLayer(inverted_model_layer)
        self.update_textbox("[END] " + self.btn_names[btn_num], "TIME_DIFF_LAST")

    def processar_segy(self):
        btn_num = 4
        fs, __ = QFileDialog.getOpenFileNames(
            QFileDialog(),
            self.btn_names[btn_num],
            self.get_cur_path(),
            self.data_types[btn_num],
        )
        self.update_textbox("[START] " + self.btn_names[btn_num])
        # fig_num = 0
        for f in fs:
            folder_name, file_name = os.path.split(f)
            self.set_cur_path(folder_name)
            self.update_textbox(file_name + "\n\tReading file", "TIME_START")
            segy = SEGY(f)
            if segy.crd_units == "decimal degrees":
                xs, ys, epsg_code = geoutm(segy.src_y, segy.src_x)
            else:
                xs = segy.src_x
                ys = segy.src_y
                # Needs to read a Location Data Stanza in the SEGY file, if it exists
                epsg_code = ""
            segy.mean = np.mean(segy.trc_data_uncorrected, axis=0)
            segy.std = np.std(segy.trc_data_uncorrected, axis=0)
            # segy.kurtosis = stats.kurtosis(segy.trc_data_uncorrected,axis=0)
            # segy.skewness = stats.skew(segy.trc_data_uncorrected,axis=0)
            enrg = segy.trc_data_uncorrected**2
            segy.enrg_mean = np.mean(enrg, axis=0)
            segy.enrg_std = np.std(enrg, axis=0)
            # segy.enrg_kurtosis = stats.kurtosis(enrg,axis=0)
            # segy.enrg_skewness = stats.skew(enrg,axis=0)
            segy.enrg_max = np.amax(enrg, axis=0)
            segy.enrg_max_2_mean = segy.enrg_max / segy.enrg_mean
            del enrg

            N = int(2 ** np.round(np.ceil(np.log2(segy.trc_data_uncorrected.shape[0]))))
            trc_data_fft = np.absolute(
                np.fft.fft(segy.trc_data_uncorrected, axis=0, n=N)
            )
            __, freqs = np.meshgrid(
                np.ones(trc_data_fft.shape[1]),
                np.fft.fftfreq(trc_data_fft.shape[0], segy.T),
            )
            fN = np.sum(freqs[:, 0] >= 0)
            trc_data_fft = np.absolute(trc_data_fft[0:fN, :]) ** 2
            freqs = freqs[0:fN, :]
            E = np.sum(trc_data_fft, axis=0)
            segy.f_cnt_enrg = np.sum(freqs * trc_data_fft, axis=0) / E
            segy.rms_bandwidth = np.sum((freqs**2) * trc_data_fft, axis=0) / E
            segy.rms_bandwidth = np.sqrt(segy.rms_bandwidth - (segy.f_cnt_enrg**2))
            segy_attr = np.array(
                [
                    10 * np.log10(segy.enrg_max),
                    20 * np.log10(np.absolute(segy.mean)),
                    segy.std,
                    10 * np.log10(segy.enrg_mean),
                    segy.enrg_std,
                    segy.enrg_max_2_mean,
                    segy.f_cnt_enrg,
                    segy.rms_bandwidth,
                ]
            )
            segy_attr_names = [
                "E_max",
                "Media",
                "Desv-pad",
                "E_media",
                "E_desv-pad",
                "E_max/media",
                "Freq_E_cnt",
                "Lrg_bnd",
            ]

            cur_traces = "%s_traces.png" % f
            cur_map_tip = '<div style="height:400;font-size:18px">\n'
            cur_map_tip += (
                '\t<div style="height:250">\n\t\t<img src="file:///%s"'
                ' style="max-height:250px" />\n\t</div>\n' % cur_traces
            )
            cur_map_tip += "</div>"

            # plt.figure(fig_num)
            plt_ext = [
                1,
                segy.n_trcs,
                (segy.delay_rec_time.max() + (segy.n_smps - 1) * segy.T) * 1000,
                (segy.delay_rec_time.min()) * 1000,
            ]
            segy.trc_data[segy.trc_data == 0] = 1e-12
            plt.matshow(
                20 * np.log10(np.absolute(segy.trc_data)),
                fignum=0,
                extent=plt_ext,
                aspect="auto",
            )
            plt.ylabel("Time (ms)")
            plt.xlabel("Trace")
            plt_clb = plt.colorbar()
            plt_clb.ax.set_ylabel("Intensity (dB)")
            plt.show()
            plt.savefig(cur_traces)
            plt.close()

            traces_layer = Create_Multipoint_Shapefile(
                "%s.shp" % f,
                xs,
                ys,
                epsg_code,
                file_name,
                segy_attr,
                segy_attr_names,
                cur_map_tip,
            )
            # traces_renderer = traces_layer.rendererV2()

            # traces_layer = QgsVectorLayer("%s.shp" % f, file_name, "ogr")
            # traces_layer.startEditing()
            # traces_layer.setMapTipTemplate(cur_map_tip)
            # traces_layer.updateFields()
            # traces_layer.updateExtents()
            # traces_layer.commitChanges()
            # print(traces_layer.commitErrors())
            # QgsVectorFileWriter.writeAsVectorFormat(traces_layer, "%s.shp" % f, "utf-8", traces_layer.crs(), 'ESRI Shapefile') #, layerOptions=["mapTipTemplate='" + cur_map_tip + "'"])
            del traces_layer
        self.update_textbox("[END] " + self.btn_names[btn_num], "TIME_DIFF_LAST")

    def carregar_segy(self):
        btn_num = 5
        fs, __ = QFileDialog.getOpenFileNames(
            QFileDialog(),
            self.btn_names[btn_num],
            self.get_cur_path(),
            self.data_types[btn_num],
        )
        self.update_textbox("[START] " + self.btn_names[btn_num])
        for f in fs:
            folder_name, file_name = os.path.split(f)
            self.set_cur_path(folder_name)
            self.update_textbox(file_name + "\n\tReading file", "TIME_START")
            if not os.path.isfile(f + ".shp"):
                continue
            segy_layer = QgsVectorLayer(f + ".shp", file_name, "ogr")

            # Remove shapefile if it is loaded
            for layer in QgsProject.instance().mapLayers().values():
                if f + ".shp" == layer.source():
                    QgsProject.instance().removeMapLayer(layer.id())

            QgsProject.instance().addMapLayer(segy_layer)

            first_attr = [l.attributes()[0] for l in segy_layer.getFeatures()]
            first_attr_name = segy_layer.getFeature(0).fields().names()[0]

            cur_traces = "%s_traces.png" % f
            cur_map_tip = '<div style="height:400;font-size:18px">\n'
            cur_map_tip += (
                '\t<div style="height:250">\n\t\t<img src="file:///%s"'
                ' style="max-height:250px" />\n\t</div>\n' % cur_traces
            )
            cur_map_tip += "</div>"
            segy_layer = Graduate_Multipoint_Shapefile(
                segy_layer, np.asarray(first_attr), first_attr_name, cur_map_tip
            )
            del segy_layer
        self.update_textbox("[END] " + self.btn_names[btn_num], "TIME_DIFF_LAST")

    def processar_simrad_db(self):
        btn_num = 6
        db = DataBinning(
            self.btn_names[btn_num],
            self.data_types[btn_num],
            parent=self,
        )
        db.exec()

    def update_textbox(self, cur_text, cur_status=""):
        if cur_status == "TIME_START":
            self.t0 = time.time()
            self.last_text = cur_text
            cur_text = self.last_text + " [START]"
        elif cur_status == "TIME_DIFF":
            cur_text0 = self.last_text + " [END] (%f s)" % (time.time() - self.t0)
            cur_text0 += "\n" + cur_text + " [START]"
            self.t0 = time.time()
            self.last_text = cur_text
            cur_text = cur_text0
        elif cur_status == "TIME_DIFF_LAST":
            cur_text0 = self.last_text + " [END] (%f s)" % (time.time() - self.t0)
            cur_text0 += "\n" + cur_text
            self.t0 = time.time()
            self.last_text = cur_text
            cur_text = cur_text0
        else:
            self.t0 = time.time()
            self.last_text = cur_text
        QgsMessageLog.logMessage("\n" + cur_text, self.plugin_title, level=Qgis.Info)
        # print(cur_text)
        # fp = open('C:\\Users\\srcae\\Desktop\\Luciano\\EchoGIS\\debug.txt', 'a')
        # fp.write(cur_text + "\n")
        # fp.close()

    def plugin_warning(self, cur_text):
        iface.messageBar().pushMessage("Error", cur_text, level=Qgis.Critical)
        # iface.messageBar().pushInfo('Status', cur_text)

    def get_cur_path(self):
        cur_settings = QSettings()
        if cur_settings.value("simrad_plugin/curpath") == None:
            return ""
        else:
            return cur_settings.value("simrad_plugin/curpath")

    def set_cur_path(self, cur_path):
        cur_settings = QSettings()
        cur_settings.setValue("simrad_plugin/curpath", cur_path)

    def get_project_CRS(self):
        self.epsg_full = iface.mapCanvas().mapSettings().destinationCrs().authid()

    def __init__(self, parent=None):
        self.plugin_title = "EchoGIS plugin"
        self.output_ext = ".jp2"
        self.output_ext = ".tif"
        self.save_wtf = True
        self.angl_corr = [
            -90,
            90,
            1,
            91,
            1,
            181,
        ]  # [min_angl_val, max_angl_val, alpha, beta, new_min_angl_val, new_max_angl_val], where 'y = alpha*x+beta'
        self.back_corr = [
            -500,
            0,
            254 / 50,
            255,
            1,
            255,
        ]  # [min_back_val, max_back_val, alpha, beta, new_min_back_val, new_max_back_val], where 'y = alpha*x+beta'
        self.new_bad_val = 0
        self.img_output_type = np.uint8
        gc.collect()
        gdal.UseExceptions()
        self.btn_names = [
            "Process .all files",
            "Load processed .all files",
            "Process shapefile for loaded .all files",
            "Load processed shapefile for loaded .all files",
            "Process .segy files",
            "Load processed .segy files",
            "Process .all files with data binning",
        ]
        self.data_types = [
            "SIMRAD files (*.all *.ALL)",
            "SIMRAD files (*.all *.ALL)",
            "SHAPEFILE files (*.shp *.SHP)",
            "SHAPEFILE files (*.shp *.SHP)",
            "SEGY files (*.seg *.SEG *.sgy *.SGY)",
            "SEGY files (*.seg *.SEG *.sgy *.SGY)",
            "SIMRAD files (*.all *.ALL)",
        ]
        self.btn_funcs = [
            self.processar_simrad,
            self.carregar_simrad,
            self.processar_shapefile,
            self.carregar_shapefile,
            self.processar_segy,
            self.carregar_segy,
            self.processar_simrad_db,
        ]
        self.get_project_CRS()
        QWidget.__init__(self, parent)
        # Create buttons
        self.layout = QVBoxLayout()
        self.plugin_buttons = []
        for b in range(len(self.btn_names)):
            self.plugin_buttons.append(QPushButton(self.btn_names[b]))
            self.plugin_buttons[b].clicked.connect(self.btn_funcs[b])
            self.layout.addWidget(self.plugin_buttons[b])
        self.setWindowTitle(self.plugin_title)
        self.setLayout(self.layout)


class SEGY:
    def __init__(self, file_name):
        file_size = os.path.getsize(file_name)
        with open(file_name, "rb") as f:
            # Textual File Header
            # dummy = struct.unpack('<'+'s'*3200, f.read(3200))
            # self.text_flhdr = ''.join(dummy)
            cur_text_flhdr = f.read(3200)
            if (
                (cur_text_flhdr[0] == 195)
                and (cur_text_flhdr[1] == 64)
                and (cur_text_flhdr[2] == 241)
            ):
                k = [
                    0,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    46,
                    46,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    46,
                    63,
                    32,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    60,
                    40,
                    43,
                    124,
                    38,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    33,
                    36,
                    42,
                    41,
                    59,
                    94,
                    45,
                    47,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    124,
                    44,
                    37,
                    95,
                    62,
                    63,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    58,
                    35,
                    64,
                    39,
                    61,
                    34,
                    46,
                    97,
                    98,
                    99,
                    100,
                    101,
                    102,
                    103,
                    104,
                    105,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    106,
                    107,
                    108,
                    109,
                    110,
                    111,
                    112,
                    113,
                    114,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    126,
                    115,
                    116,
                    117,
                    118,
                    119,
                    120,
                    121,
                    122,
                    46,
                    46,
                    46,
                    91,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    93,
                    46,
                    46,
                    123,
                    65,
                    66,
                    67,
                    68,
                    69,
                    70,
                    71,
                    72,
                    73,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    125,
                    74,
                    75,
                    76,
                    77,
                    78,
                    79,
                    80,
                    81,
                    82,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    92,
                    46,
                    83,
                    84,
                    85,
                    86,
                    87,
                    88,
                    89,
                    90,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    46,
                    46,
                    46,
                    46,
                    46,
                    46,
                ]
                self.text_flhdr = [chr(k[i]) for i in cur_text_flhdr]
                self.text_flhdr = "".join(self.text_flhdr)
                is_ebcdic = 1
            else:
                self.text_flhdr = "".join(
                    [
                        i.decode()
                        for i in struct.unpack("<" + "s" * 3200, cur_text_flhdr)
                    ]
                )
                is_ebcdic = 0
            self.text_flhdr = [
                self.text_flhdr[i : (i + 80)] + "\n" for i in range(0, 3200, 80)
            ]
            self.text_flhdr = "".join(self.text_flhdr)

            # Binary File Header
            field_names = (
                "job_n",
                "line_n",
                "reel_n",
                "n_data_trc",
                "n_aux_trc",
                "smp_int",
                "smp_int_org_fld_rec",
                "n_smp_trc",
                "n_smp_trc_org_fld_rec",
                "data_smp_fmt_code",
                "ensmb_fold",
                "trc_sort_code",
                "vert_sum_code",
                "swp_freq_start",
                "swp_freq_stop",
                "swp_len",
                "swp_type",
                "trc_n_swp_chn",
                "swp_tpr_lng_start",
                "swp_tpr_lng_stop",
                "tpr_type",
                "corr_data_tr",
                "bin_gain_rec",
                "amp_rec_met",
                "meas_sys",
                "imp_sig_pol",
                "vib_pol_code",
            )  # , 'ext_n_data_trc', 'ext_n_aux_trc', 'ext_smp_int')
            field_types = "IIIHHHHHHHHHHHHHHHHHHHHHHHH"  # IId'
            alignment_type = ">"
            self.bin_flhdr, cur_index = read_initial_datagram_bytes(
                f.read(400), field_names, field_types, alignment_type
            )

            self.trc_hdr = []
            self.delay_rec_time = []
            self.n_smps = self.bin_flhdr["n_smp_trc"]
            self.n_trcs = int((file_size - 3600) / (240 + self.n_smps * 4) + 0.5)
            self.trc_data = np.zeros((int(self.n_trcs), self.n_smps))
            self.src_x = np.zeros(int(self.n_trcs))
            self.src_y = np.zeros(int(self.n_trcs))
            self.crd_units = self.bin_flhdr["meas_sys"]
            trc_n = 0
            while f.tell() < file_size:
                field_names = (
                    "trc_seqn_line",
                    "trc_seqn_segy",
                    "org_fld_recn",
                    "trcn_wthn_org_fld_rec",
                    "enrg_src_ptn",
                    "ens_n",
                    "trcn_ens",
                    "trc_id_code",
                    "nvrt_sum_trc",
                    "nhrz_stc_trc",
                    "data_use",
                    "dist_cntrc_src_rcv",
                    "rcv_grp_elv",
                    "srf_evl_src",
                    "src_dpt_blw_srf",
                    "dtm_elv_rcv_grp",
                    "dtm_elv_src",
                    "wtr_dpt_src",
                    "wtr_dpt_grp",
                    "scl_bytes_41_68",
                    "scl_bytes_73_88",
                    "src_x",
                    "src_y",
                    "grp_x",
                    "grp_y",
                    "crd_units",
                    "wtr_vlc",
                    "sub_wtr_vlc",
                    "uph_ms_src",
                    "uph_ms_grp",
                    "src_stc_crr_ms",
                    "grp_stc_crr_ms",
                    "tot_stc",
                    "lag_time_a",
                    "lag_time_b",
                    "delay_rec_ms",
                    "mute_time_str",
                    "mute_time_stp",
                    "nsmp_trc",
                    "smp_int_us_trc",
                    "gain_type_fld_inst",
                    "inst_gain_cns_db",
                    "inst_init_gain_db",
                    "corr",
                    "swp_frq_str_trc",
                    "swp_frq_stp_trc",
                    "swp_lng_trc",
                    "swp_type_trc",
                    "swp_tpr_lng_str_trc",
                    "swp_tpr_lng_stp_trc",
                    "tpr_type_trc",
                    "alias_flt_frq",
                    "alias_flt_slp",
                    "notch_flt_frq",
                    "notch_flt_slp",
                    "lo_cut_flt_frq",
                    "hi_cut_flt_frq",
                    "lo_cut_flt_slp",
                    "hi_cut_flt_slp",
                    "year_trc",
                    "day_trc",
                    "hour_trc",
                    "minute_trc",
                    "second_trc",
                    "time_basis_code",
                    "trc_wgt_fact",
                    "geop_grpn_rll_swt_pos1",
                    "geop_grpn_trcn1",
                    "geop_grpn_lst_trc",
                    "gap_size",
                    "ovr_trv",
                    "cdp_x",
                    "cdp_y",
                    "poststack_3d_1",
                    "poststack_3d_2",
                    "shotpointn",
                    "scl_shotpointn",
                    "trc_value_meas_unit",
                    "transduc_cns",
                    "transduc_units",
                    "trc_id",
                    "scl_time_trc",
                    "src_type",
                    "src_enrg_dir1",
                    "src_enrg_dir2",
                    "src_enrg_dir3",
                    "src_meas_mant",
                    "src_meas_exp",
                    "src_meas_unit",
                )
                field_types = "IIIIIIIHHHHIIIIIIIIhhiiiiHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHIIIIIHHIHHHHHHHIHI"
                alignment_type = ">"
                cur_trc_hdr, cur_index = read_initial_datagram_bytes(
                    f.read(240), field_names, field_types, alignment_type
                )
                self.delay_rec_time.append(cur_trc_hdr["delay_rec_ms"])
                self.trc_hdr.append(cur_trc_hdr)
                cur_trc, cur_index = read_datagram_array(
                    f.read(self.bin_flhdr["n_smp_trc"] * 4),
                    self.bin_flhdr["n_smp_trc"],
                    "trcs",
                    "f",
                    alignment_type,
                )
                self.trc_data[trc_n][:] = np.asarray(cur_trc["trcs"])
                # if trc_n<10:
                #     print((cur_trc_hdr['wtr_dpt_grp'],cur_trc_hdr['scl_bytes_41_68'],cur_trc_hdr['scl_bytes_73_88']))
                if trc_n == 0:
                    if cur_trc_hdr["crd_units"] == 1:
                        if self.crd_units == 1:
                            self.crd_units = "meters"
                        else:
                            self.crd_units = "feet"
                    else:
                        self.crd_units = "decimal degrees"
                if cur_trc_hdr["scl_bytes_73_88"] < 0:
                    self.src_x[trc_n] = float(cur_trc_hdr["src_x"]) / np.absolute(
                        float(cur_trc_hdr["scl_bytes_73_88"])
                    )
                    self.src_y[trc_n] = float(cur_trc_hdr["src_y"]) / np.absolute(
                        float(cur_trc_hdr["scl_bytes_73_88"])
                    )
                else:
                    self.src_x[trc_n] = float(cur_trc_hdr["src_x"]) * float(
                        cur_trc_hdr["scl_bytes_73_88"]
                    )
                    self.src_y[trc_n] = float(cur_trc_hdr["src_y"]) * float(
                        cur_trc_hdr["scl_bytes_73_88"]
                    )
                # if cur_trc_hdr['crd_units']==1: # Meters or feet
                if cur_trc_hdr["crd_units"] == 2:  # Arc seconds
                    self.src_x[trc_n] /= 3600.0
                    self.src_y[trc_n] /= 3600.0
                # elif cur_trc_hdr['crd_units']==3: # Decimal degrees
                elif (
                    cur_trc_hdr["crd_units"] == 4
                ):  # DMS (degrees, minutes and seconds)
                    dummy = self.src_x[trc_n]
                    decimal_place = np.floor(dummy / 10000.0)
                    self.src_x[trc_n] = decimal_place
                    dummy -= decimal_place * 10000.0
                    decimal_place = np.floor(dummy / 100.0)
                    self.src_x[trc_n] += decimal_place / 60.0
                    dummy -= decimal_place * 100.0
                    self.src_x[trc_n] += np.floor(dummy) / 3600.0 + np.remainder(dummy)
                    dummy = self.src_y[trc_n]
                    decimal_place = np.floor(dummy / 10000.0)
                    self.src_y[trc_n] = decimal_place
                    dummy -= decimal_place * 10000.0
                    decimal_place = np.floor(dummy / 100.0)
                    self.src_y[trc_n] += decimal_place / 60.0
                    dummy -= decimal_place * 100.0
                    self.src_y[trc_n] += np.floor(dummy) / 3600.0 + np.remainder(dummy)
                # print("%f %f %f" % (cur_trc_hdr['scl_bytes_73_88'], cur_trc_hdr['src_x'], cur_trc_hdr['src_y']))
                trc_n += 1
            self.trc_data = self.trc_data.T
            self.delay_rec_time = (np.asarray(self.delay_rec_time).astype(float)) / 1000
            self.T = float(self.bin_flhdr["smp_int"]) * 1e-6
            un_delay_rec_time = np.unique(self.delay_rec_time)
            self.trc_data_uncorrected = self.trc_data
            if len(un_delay_rec_time) > 1:
                extra_smps = np.round(
                    (self.delay_rec_time - un_delay_rec_time[0]) / self.T
                ).astype(int)
                max_extra_smps = extra_smps.max()
                ext_trc_data = np.zeros(
                    (int(self.n_smps + extra_smps.max()), self.n_trcs)
                )
                for n in range(self.n_trcs):
                    ext_trc_data[
                        (extra_smps[n]) : (self.n_smps + extra_smps[n]), n
                    ] = self.trc_data[:, n]
                self.trc_data = ext_trc_data


class DataBinning(QDialog):
    def __init__(self, name, data_type, parent=None):
        super().__init__(parent=parent)
        self.plugin_title = "EchoGIS plugin"

        self.name = name
        self.data_type = data_type

        pixel_size_validator = QDoubleValidator(1.0, 250.0, 2)
        pixel_size_validator.setNotation(QDoubleValidator.StandardNotation)

        self.layout = QVBoxLayout()
        self.upper_layout1 = QHBoxLayout()
        self.fieldLabel1 = QLabel()
        self.fieldLabel1.setText("Pixel size BM: 1-250 [m]")
        self.fieldLabel1.setMinimumWidth(180)
        self.upper_layout1.addWidget(self.fieldLabel1)
        self.field1 = QLineEdit()
        # self.field1.setObjectName("Pixel size BM: 1-100 (metros)")
        self.field1.setText("25")
        self.field1.setValidator(pixel_size_validator)
        self.field1.setMaxLength(8)
        self.field1.setMaximumWidth(60)
        self.upper_layout1.addWidget(self.field1)
        self.layout.addLayout(self.upper_layout1, 1)

        self.upper_layout2 = QHBoxLayout()
        self.fieldLabel2 = QLabel()
        self.fieldLabel2.setText("Pixel size BS: 1-250 [m]")
        self.fieldLabel2.setMinimumWidth(180)
        self.upper_layout2.addWidget(self.fieldLabel2)
        self.field2 = QLineEdit()
        # self.field2.setObjectName("Pixel size BS: 1-250 [m]")
        self.field2.setText("10")
        self.field2.setValidator(pixel_size_validator)
        self.field2.setMaxLength(8)
        self.field2.setMaximumWidth(60)
        self.upper_layout2.addWidget(self.field2)
        self.layout.addLayout(self.upper_layout2, 1)

        self.itplcheck = QCheckBox("Interpolate data to fill holes")
        self.itplcheck.setChecked(True)
        # self.itplcheck.setToolTip(
        #     "Aplica dilatação e interpola na área dilatada. "
        #     + 'É uma tentativa de se preencher "buracos".'
        # )
        self.layout.addWidget(self.itplcheck)

        self.processBtn = QPushButton("Process")
        self.processBtn.clicked.connect(self.processar)
        self.layout.addWidget(self.processBtn)

        self.setWindowTitle("Data binning")
        self.setLayout(self.layout)

    def processar(self):
        fs = QFileDialog.getOpenFileNames(
            QFileDialog(),
            self.name,
            self.get_cur_path(),
            self.data_type,
        )[0]

        if not fs:
            self.update_textbox("Nenhum arquivo lido!")
            return

        self.update_textbox("Processamento iniciado")

        bad_val = -30000

        solar_vector = np.asarray([1, 1, 1])

        # Validações
        interp = self.itplcheck.isChecked()

        pixel_size_bm = self.receive_input(
            self.field1.text(), "Tamanho de pixel inválido para BM!"
        )
        pixel_size_bs = self.receive_input(
            self.field2.text(), "Tamanho de pixel inválido para BS!"
        )

        if not pixel_size_bm or not pixel_size_bs:
            return

        self.update_textbox(
            "Tamanhos de pixel: BM={:.2f}, BS={:.2f}".format(
                pixel_size_bm, pixel_size_bs
            ),
            False,
        )

        if interp:
            self.update_textbox("Será realizada interpolação na área dilatada.", False)

        position = []
        depth = []
        layer_names = [
            "Retroespalhamento",
            "Batimetria",
            # "Relevo de batimetria",
        ]
        layer_ranges = []
        bm_layers = []
        bm_ranges = []
        rel_bm_layers = []

        # Grades de batimetria
        self.update_textbox("Lendo arquivos")
        for f in fs:
            folder_name, file_name = os.path.split(f)
            self.set_cur_path(folder_name)
            self.update_textbox("Lendo arquivo " + file_name)
            bm_layers.append("bm_" + file_name)
            rel_bm_layers.append("rel_bm_" + file_name)
            pos_dgs, depth_dgs, model = read_pos_dep_datagrams_db(f)
            position.append(pos_dgs)
            depth.append(depth_dgs)
            gc.collect()

        if model == 850:
            self.update_textbox("Modelo: ME70BO", False)
        elif model == 2045:
            self.update_textbox("Modelo: EM2040C", False)
        else:
            self.update_textbox("Modelo: EM{}".format(model), False)

        position, depth, epsg_code = interpolate_data_db(position, depth)
        self.update_textbox(epsg_code, False)

        self.update_textbox("Criando grades de batimetria")
        grid, info = create_grid_db(depth, pixel_size_bm, bad_val)
        gc.collect()
        grid["bm"] = ndimage.median_filter(grid["bm"], size=3)
        if interp:
            grid["bm"] = interpolate_grid_db(grid["bm"], pixel_size_bm, bad_val)
            # for i, layer in enumerate(bm_layers):
            #     grid_bm = create_bm_grid_db(depth[i], info, pixel_size_bm, bad_val)
            #     min_val = np.amin(grid_bm[grid_bm > bad_val])
            #     max_val = np.amax(grid_bm)
            #     bm_ranges.append((min_val, max_val))
            #     if interp:
            #         grid_bm = interpolate_grid_db(grid_bm, pixel_size_bm, bad_val)
            #     save_np_array_db(
            #         grid_bm.astype(np.float32),
            #         "{}/{}.tif".format(folder_name, layer),
            #     )
            # grid_bm_sol = dot_solar_vector_db(grid_bm, solar_vector, bad_val)
            # grid_bm_rel = generate_rgb_db(
            #     normalize_grid_db(grid_bm, bad_val),
            #     normalize_grid_db(grid_bm_sol, bad_val),
            #     bad_val,
            # )
            # grid_bm_rel = (grid_bm_rel * 255).astype(np.uint8)
            # save_np_array_rgb_db(
            #     grid_bm_rel, "{}/{}.tif".format(folder_name, rel_bm_layers[i])
            # )
            gc.collect()

        self.update_textbox("Criando grades de retroespalhamento")
        grid["sid"] = []

        run_checksum = False

        for i, f in enumerate(fs):
            # tracemalloc.start(25)
            # Split file and folder names
            folder_name, file_name = os.path.split(f)
            self.set_cur_path(folder_name)
            self.update_textbox2(file_name + "\n\tReading file", "TIME_START")
            _, sid_dgs, _, rra_dgs, run_dgs = read_datagrams(f, run_checksum)
            gc.collect()
            self.update_textbox(
                file_name + "\n\tArranging datagrams to create waterfall", "TIME_DIFF"
            )
            # Get pings for SID, XYZ and RRA datagrams
            xyz_dgs = depth[i]
            png_sid = np.asarray([dg["ping"] for dg in sid_dgs])
            png_xyz = np.asarray([dg["ping"] for dg in xyz_dgs])
            png_rra = np.asarray([dg["ping"] for dg in rra_dgs])
            if png_sid.size == 0:
                self.plugin_warning(
                    file_name
                    + ': file does not contain datagrams for "seabed image data 89"'
                )
                self.plugin_warning(".")
            if png_rra.size == 0:
                self.plugin_warning(
                    file_name
                    + ': file does not contain datagrams for "raw range and angle 78"'
                )
                self.plugin_warning(".")
            # Find matching pings for SID, XYZ and RRA datagrams
            png_unq = np.intersect1d(np.intersect1d(png_sid, png_xyz), png_rra)
            sid_dgs = [
                sid_dgs[k]
                for k in np.intersect1d(png_sid, png_unq, return_indices=True)[1]
            ]
            rra_dgs = [
                rra_dgs[k]
                for k in np.intersect1d(png_rra, png_unq, return_indices=True)[1]
            ]
            xyz_dgs = [
                xyz_dgs[k]
                for k in np.intersect1d(png_xyz, png_unq, return_indices=True)[1]
            ]
            # Sort datagrams by time
            ts = get_datenum(sid_dgs)
            if ts != sorted(ts):
                tsi = np.argsort(ts)
                sid_dgs = [sid_dgs[i] for i in tsi]
                rra_dgs = [rra_dgs[i] for i in tsi]
                xyz_dgs = [xyz_dgs[i] for i in tsi]
                del tsi
            del ts
            gc.collect()
            self.update_textbox2(file_name + "\n\tCreating waterfall", "TIME_DIFF")

            grid_sid = create_sid_grids_db(
                sid_dgs, xyz_dgs, rra_dgs, info, pixel_size_bs, bad_val
            )
            # grid["sid"].append(ndimage.median_filter(grid_sid, size=3))
            grid["sid"].append(grid_sid)

        # # Grades de retroespalhamento
        # for i, f in enumerate(fs):
        #     folder_name, file_name = os.path.split(f)
        #     sid_dgs = read_seabed_datagrams_db(f)
        #     sid_dgs = sort_sid_db(sid_dgs)
        #     grid_sid = create_sid_grids_db(
        #         sid_dgs, depth[i], info, pixel_size_bs, bad_val
        #     )
        #     grid["sid"].append(grid_sid)
        #     gc.collect()

        grid["sid"] = create_sid_grid_db(grid["sid"], bad_val)
        if interp:
            grid["sid"] = interpolate_grid_db(grid["sid"], pixel_size_bs, bad_val)
        gc.collect()

        # Ajuste de imagem
        # self.update_textbox("Criando grade de batimetria com relevo")
        # grid["bm_solar"] = dot_solar_vector_db(grid["bm"], solar_vector, bad_val)
        # grid["bm_RGB"] = generate_rgb_db(
        #     normalize_grid_db(grid["bm"].copy(), bad_val),
        #     normalize_grid_db(grid["bm_solar"].copy(), bad_val),
        #     bad_val,
        # )
        gc.collect()

        # layer_names.extend(bm_layers)
        # layer_names.extend(rel_bm_layers)

        self.update_textbox("Ajustando as imagens")
        bs_img = grid["sid"].astype(np.float32)
        bt_img = grid["bm"].astype(np.float32)
        # bt_img_rgb = (grid["bm_RGB"] * 255).astype(np.uint8)

        self.update_textbox("Salvando as imagens")
        save_np_array_db(bs_img, "{}/{}.tif".format(folder_name, layer_names[0]))
        save_np_array_db(bt_img, "{}/{}.tif".format(folder_name, layer_names[1]))
        # save_np_array_rgb_db(
        #     bt_img_rgb, "{}/{}.tif".format(folder_name, layer_names[2])
        # )
        gc.collect()

        # Carregar no QGIS
        remove_layers_db(layer_names)

        self.update_textbox("Georreferenciando")
        gcpList = [
            get_convex_control_points_db(
                depth, bs_img, bad_val, info
            ),  # retroespalhamento
            get_convex_control_points_db(depth, bt_img, bad_val, info),
        ]  # batimetria
        gc.collect()

        self.update_textbox("Salvando imagens georreferenciadas")
        for layer in layer_names[::-1]:
            img_tiff = "{}/{}.tif".format(folder_name, layer)
            geo_tiff = "{}/{}_geo.tif".format(folder_name, layer)
            if layer == layer_names[0]:
                translate_warp_db(img_tiff, gcpList[0], epsg_code, geo_tiff)
            else:
                translate_warp_db(img_tiff, gcpList[1], epsg_code, geo_tiff)
            # os.remove(img_tiff)

        self.update_textbox("Carregando imagens georreferenciadas")
        min_val_bs = -55  # np.amin(bs_img[bs_img > bad_val])
        max_val_bs = 0  # np.amax(bs_img)
        layer_ranges.append((min_val_bs, max_val_bs))
        mean_bs = np.mean(bs_img[bs_img > bad_val])
        std_bs = np.std(bs_img[bs_img > bad_val])
        min_val_bm = np.amin(bt_img[bt_img > bad_val])
        max_val_bm = np.amax(bt_img)
        layer_ranges.append((min_val_bm, max_val_bm))
        layer_ranges.append((min_val_bm, max_val_bm))
        mean_bm = np.mean(bt_img[bt_img > bad_val])
        std_bm = np.std(bt_img[bt_img > bad_val])
        layer_ranges.extend(bm_ranges)

        self.update_textbox("Estatísticas das grades totais:", False)
        self.update_textbox(
            "   Valor min BS: {:.2f}; Valor max BS: {:.2f}".format(
                min_val_bs, max_val_bs
            ),
            False,
        )
        self.update_textbox(
            "   Valor médio BS: {:.2f}; Desvio padrão BS {:.2f}".format(
                mean_bs, std_bs
            ),
            False,
        )
        self.update_textbox(
            "   Valor min BM: {:.2f}; Valor max BM: {:.2f}".format(
                min_val_bm, max_val_bm
            ),
            False,
        )
        self.update_textbox(
            "   Valor médio BM: {:.2f}; Desvio padrão BM {:.2f}".format(
                mean_bm, std_bm
            ),
            False,
        )
        # print(layer_ranges)
        for i, layer in enumerate(layer_names):
            self.update_textbox("Carregando " + layer)
            if layer == layer_names[0]:
                load_image_QGIS_db(
                    "{}/{}_geo.tif".format(folder_name, layer),
                    layer,
                    layer_ranges[i],
                )
            elif layer == layer_names[1] or layer[0:2] == "bm":
                load_image_QGIS_db(
                    "{}/{}_geo.tif".format(folder_name, layer),
                    layer,
                    layer_ranges[i],
                    False,
                )
            else:
                load_multiband_image_QGIS_db(
                    "{}/{}_geo.tif".format(folder_name, layer), layer
                )
            gc.collect()

        self.update_textbox("Processar .all: completo")

    def update_textbox(self, cur_text, status=True):
        if status:
            # QGIS 3.2
            iface.messageBar().pushInfo("Status", cur_text)

            cur_time = time.strftime("[%H:%M:%S] ")
            print(cur_time + cur_text + "\n")
        else:
            print("  " + cur_text + "\n")
        QApplication.processEvents()

    def update_textbox2(self, cur_text, cur_status=""):
        if cur_status == "TIME_START":
            self.t0 = time.time()
            self.last_text = cur_text
            cur_text = self.last_text + " [START]"
        elif cur_status == "TIME_DIFF":
            cur_text0 = self.last_text + " [END] (%f s)" % (time.time() - self.t0)
            cur_text0 += "\n" + cur_text + " [START]"
            self.t0 = time.time()
            self.last_text = cur_text
            cur_text = cur_text0
        elif cur_status == "TIME_DIFF_LAST":
            cur_text0 = self.last_text + " [END] (%f s)" % (time.time() - self.t0)
            cur_text0 += "\n" + cur_text
            self.t0 = time.time()
            self.last_text = cur_text
            cur_text = cur_text0
        else:
            self.t0 = time.time()
            self.last_text = cur_text
        QgsMessageLog.logMessage("\n" + cur_text, self.plugin_title, level=Qgis.Info)

    def get_cur_path(self):
        cur_settings = QSettings()
        if cur_settings.value("simrad_plugin/curpath") == None:
            return ""
        else:
            return cur_settings.value("simrad_plugin/curpath")

    def set_cur_path(self, cur_path):
        cur_settings = QSettings()
        cur_settings.setValue("simrad_plugin/curpath", cur_path)

    def receive_input(self, inp_val, error_msg):
        try:
            flt_val = float(inp_val)
        except:
            self.update_textbox(error_msg)
            return False
        if flt_val < 1 or flt_val > 100:
            self.update_textbox(error_msg)
            return False
        return flt_val


class DataBinningHist:
    def __init__(
        self,
        rows: int,
        cols: int,
        min_val: float = 0,
        max_val: float = 1,
        n_bins: int = 100,
        min_smps: int = 3,
        no_val: float = -1,
    ):
        self.rows = rows
        self.cols = cols
        self.min_val = min_val
        self.max_val = max_val
        self.n_bins = n_bins
        self.min_spms = min_smps
        self.no_val = no_val

        self.hists = np.zeros((self.rows, self.cols, self.n_bins))
        self.vals = np.linspace(self.min_val, self.max_val, self.n_bins, endpoint=False)

    def add_value(self, row: int, col: int, value: float):
        idx = self.map_value(value)
        self.hists[row, col, idx] += 1

    def map_value(self, value: float) -> int:
        if value < self.min_val:
            value = self.min_val
        elif value > self.max_val:
            value = self.max_val

        idx = int(self.n_bins * (value - self.min_val) / (self.max_val - self.min_val))

        if idx >= self.n_bins:
            idx -= 1

        return idx

    def calc_median(self) -> np.ndarray:
        median = np.ones((self.rows, self.cols)) * self.no_val

        for row in range(self.rows):
            for col in range(self.cols):
                N = self.hists[row, col].sum()
                if N > self.min_spms:
                    median_idx = (N + 1) / 2
                    total = 0
                    for i, count in enumerate(self.hists[row, col]):
                        total += count
                        if total > median_idx:
                            median[row, col] = self.vals[i]
                            break

        return median


def read_datagrams(file_name, run_checksum):
    file_size = os.path.getsize(file_name)
    pos_dgs = []
    sid_dgs = []
    xyz_dgs = []
    rra_dgs = []
    run_dgs = []
    with open(file_name, "rb") as f:
        while f.tell() < file_size:
            cur_dg_len = struct.unpack("I", f.read(4))[0]
            cur_dg_id = struct.unpack("cc", f.read(2))[1]
            cur_dg = f.read(cur_dg_len - 2)
            if run_checksum and checksum_error(cur_dg, cur_dg_id) == False:
                print("Bad checksum for %s datagram" % cur_dg_id)
            if cur_dg_id == b"P":
                pos_dgs.append(read_position_datagrams(cur_dg))
            elif cur_dg_id == b"Y":
                sid_dgs.append(read_sid_89_datagrams(cur_dg))
            elif cur_dg_id == b"X":
                xyz_dgs.append(read_xyz_88_datagrams(cur_dg))
            elif cur_dg_id == b"N":
                rra_dgs.append(read_rra_78_datagrams(cur_dg))
            elif cur_dg_id == b"R":
                run_dgs.append(read_runtime_param_datagrams(cur_dg))
    return pos_dgs, sid_dgs, xyz_dgs, rra_dgs, run_dgs


def checksum_error(cur_bytes, cur_dg_id):
    dg_checksum = struct.unpack("<H", cur_bytes[-2:])[0]
    cur_bytes = cur_bytes[:-3]
    cur_bytes_sum = sum(struct.unpack("<" + "B" * len(cur_bytes), cur_bytes))
    cur_bytes_sum += struct.unpack("<B", cur_dg_id)[0]
    # print(cur_dg_id)
    # print(cur_bytes_sum)
    return (cur_bytes_sum & 65535) == dg_checksum


def read_initial_datagram_bytes(cur_bytes, field_names, field_types, alignment_type):
    field_types = alignment_type + field_types
    cur_index = struct.calcsize(field_types)
    field_values = struct.unpack(field_types, cur_bytes[:cur_index])
    cur_dg = dict(zip(field_names, field_values))
    return cur_dg, cur_index


def read_cycled_datagram_bytes(
    cur_bytes, cycle_len, field_names, field_types, alignment_type
):
    field_types = alignment_type + field_types * cycle_len
    cur_index = struct.calcsize(field_types)
    field_values = struct.unpack(field_types, cur_bytes[:cur_index])
    cur_dg = {}
    N = len(field_names)
    for k in range(N):
        cur_dg[field_names[k]] = field_values[k::N]
    return cur_dg, cur_index


def read_datagram_array(cur_bytes, array_len, field_name, field_type, alignment_type):
    field_type = alignment_type + field_type * array_len
    cur_index = struct.calcsize(field_type)
    cur_dg = {}
    cur_dg[field_name] = struct.unpack(field_type, cur_bytes[:cur_index])
    return cur_dg, cur_index


def read_position_datagrams(cur_bytes):
    field_names = (
        "em_model",
        "date",
        "time",
        "pos_counter",
        "system_sn",
        "latitude",
        "longitude",
        "measure_fix_qual_cm",
        "speed_cm_s",
        "course_dot01",
        "heading_dot01",
        "pos_system_descriptor",
        "pos_in_dg_N",
    )
    field_types = "HIIHHiiHHHHBB"
    alignment_type = "<"
    cur_dg, cur_index = read_initial_datagram_bytes(
        cur_bytes, field_names, field_types, alignment_type
    )
    cur_dg["pos_in_dg"] = cur_bytes[cur_index:-3]
    cur_dg["latitude"] = float(cur_dg["latitude"]) / 20000000
    cur_dg["longitude"] = float(cur_dg["longitude"]) / 10000000
    return cur_dg


def read_runtime_param_datagrams(cur_bytes):
    field_names = (
        "em_model",
        "date",
        "time",
        "ping_counter",
        "system_sn",
        "operator_station_status",
        "CPU_status",
        "BSP_status",
        "sonar_head_status",
        "mode",
        "filter_id",
        "min_depth",
        "max_depth",
        "absorb_coeff",
        "transmit_pulse_len",
        "transmit_beamwidth",
        "transmit_power_re_max",
        "receive_beamwidth",
        "receive_bandwidth",
        "receiver_fixed_gain_setting",
        "TVG_law_crossover_ang",
        "src_soundspeed_transducer",
        "max_port_swath_width",
        "beam_spacing",
        "max_port_coverage",
        "yaw_pitch_stab_mode",
        "max_starboard_coverage",
        "max_starboard_swath_width",
        "transmit_along_tilt",
        "filter_id2",
    )
    field_types = "HIIHHBBBBBBHHHHHbBBBBBHBBBBHhB"
    alignment_type = "<"
    cur_dg, cur_index = read_initial_datagram_bytes(
        cur_bytes, field_names, field_types, alignment_type
    )
    return cur_dg


def read_sid_89_datagrams(cur_bytes):
    field_names = (
        "em_model",
        "date",
        "time",
        "ping",
        "system_sn",
        "smp_freq",
        "range_norm_inc",
        "bsn",
        "obs",
        "tx_beamwidth",
        "tvg_law",
        "N",
    )
    field_types = "HIIHHfHhhHHH"
    alignment_type = "<"
    cur_dg, cur_index1 = read_initial_datagram_bytes(
        cur_bytes, field_names, field_types, alignment_type
    )
    field_names = ("sort_dir", "det_info", "Ns", "cnt_num")
    field_types = "bBHH"
    alignment_type = "<"
    cur_extra_dg, cur_index2 = read_cycled_datagram_bytes(
        cur_bytes[cur_index1:], cur_dg["N"], field_names, field_types, alignment_type
    )
    cur_dg.update(cur_extra_dg)
    cur_extra_dg, cur_index3 = read_datagram_array(
        cur_bytes[(cur_index1 + cur_index2) :], sum(cur_dg["Ns"]), "samples", "h", "<"
    )
    cur_dg.update(cur_extra_dg)
    # cur_dg['samples'] = [float(x)/10 for x in cur_dg['samples']]
    return cur_dg


def read_xyz_88_datagrams(cur_bytes):
    field_names = (
        "em_model",
        "date",
        "time",
        "ping",
        "system_sn",
        "heading_dot01",
        "sound_speed_dm_s",
        "depth_re_water_level",
        "N",
        "N_valid",
        "smp_freq",
        "scanning_info",
    )
    field_types = "HIIHHHHfHHfB"
    alignment_type = "<"
    cur_dg, cur_index = read_initial_datagram_bytes(
        cur_bytes, field_names, field_types, alignment_type
    )
    cur_index += 3  # Spare bytes
    field_names = (
        "z",
        "y",
        "x",
        "detect_win_len",
        "quality_factor",
        "beam_inc_angle_adjust",
        "detect_info",
        "real_time_cleaning_info",
        "reflectivity_dot1dB",
    )
    field_types = "fffHBbBbh"
    alignment_type = "<"
    cur_extra_dg, cur_index = read_cycled_datagram_bytes(
        cur_bytes[cur_index:], cur_dg["N"], field_names, field_types, alignment_type
    )
    cur_dg.update(cur_extra_dg)
    return cur_dg


def read_rra_78_datagrams(cur_bytes):
    field_names = (
        "em_model",
        "date",
        "time",
        "ping",
        "system_sn",
        "soundspeed",
        "Ntx",
        "Nrx",
        "N_valid_det",
        "smp_freq",
        "dscale",
    )
    field_types = "HIIHHHHHHfI"
    alignment_type = "<"
    cur_dg, cur_index1 = read_initial_datagram_bytes(
        cur_bytes, field_names, field_types, alignment_type
    )
    field_names = (
        "tilt_angle",
        "focus_range",
        "signal_len",
        "transmit_dly",
        "centre_freq",
        "mean_absorption",
        "waveform_id",
        "tx_array_ind",
        "bandwidth",
    )
    field_types = "hHfffHBBf"
    alignment_type = "<"
    cur_extra_dg, cur_index2 = read_cycled_datagram_bytes(
        cur_bytes[cur_index1:], cur_dg["Ntx"], field_names, field_types, alignment_type
    )
    cur_dg.update(cur_extra_dg)
    field_names = (
        "beam_point_angle",
        "tx_sec_num",
        "detect_info",
        "detect_win_len",
        "qual_fact",
        "dcorr",
        "two_way_travel",
        "reflectivity",
        "realtime_clean_info",
        "spare",
    )
    field_types = "hBBHBbfhbB"
    alignment_type = "<"
    cur_extra_dg, cur_index3 = read_cycled_datagram_bytes(
        cur_bytes[(cur_index1 + cur_index2) :],
        cur_dg["Nrx"],
        field_names,
        field_types,
        alignment_type,
    )
    cur_dg.update(cur_extra_dg)
    del cur_dg["spare"]
    return cur_dg


def get_waterfall_info(sid_dg, xyz_dg, rra_dg):
    # Read samples and center sample information
    # (position in vector of samples, y values
    # and two-way travel times)
    smps = (
        np.asarray(sid_dg["samples"], dtype=np.float32) / 10
    )  # Resolution: 0.1dB, multiplied by 10
    xs = np.asarray(xyz_dg["x"], dtype=np.float32)
    ys = np.asarray(xyz_dg["y"], dtype=np.float32)
    zs = np.asarray(xyz_dg["z"], dtype=np.float32)
    travel_times = np.asarray(rra_dg["two_way_travel"], dtype=np.float32)
    travel_times = travel_times / 2
    Ns = np.asarray(sid_dg["Ns"])
    cnt_smps = np.cumsum(Ns[0:-1])
    cnt_smps = np.insert(cnt_smps, 0, 0)
    cnt_smps = np.add(cnt_smps, sid_dg["cnt_num"]) - 1
    smp_freq = float(sid_dg["smp_freq"])
    reflectivity = (
        np.asarray(xyz_dg["reflectivity_dot1dB"], dtype=np.float32) / 10
    )  # Resolution: 0.1dB, multiplied by 10
    sound_speed = float(xyz_dg["sound_speed_dm_s"]) / 10
    ob_bs = float(sid_dg["obs"]) / 10.0
    ni_bs = float(sid_dg["bsn"]) / 10.0
    Ro = float(sid_dg["range_norm_inc"])
    # Sort all information by the center samples' y values
    ysi = np.argsort(ys)
    xs = xs[ysi]
    ys = ys[ysi]
    zs = zs[ysi]
    travel_times = travel_times[ysi]
    cnt_smps = cnt_smps[ysi]
    Ns = Ns[ysi]
    reflectivity = reflectivity[ysi]
    return (
        smps,
        xs,
        ys,
        zs,
        travel_times,
        cnt_smps,
        reflectivity,
        Ns,
        smp_freq,
        sound_speed,
        ob_bs,
        ni_bs,
        Ro,
    )


def outside_hull(test_points, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(test_points) < 0


def show_pts(x, y, plt_title):
    plt.figure()
    plt.scatter(x, y)
    plt.grid()
    plt.title(plt_title)
    plt.show()


def create_waterfall(sid_dgs, xyz_dgs, rra_dgs, pixel_size, run_dgs):
    no_val = -30000
    max_y = max(
        np.asarray(
            [np.ceil(max(np.absolute(xyz["y"])) / pixel_size) for xyz in xyz_dgs]
        )
    )
    W = max_y.astype(int) * 2 + 1

    pulse_width = float(run_dgs[0]["transmit_pulse_len"]) * 1e-6
    transmit_beamwidth = float(run_dgs[0]["transmit_beamwidth"]) / 10.0 * math.pi / 180
    receive_beamwidth = float(run_dgs[0]["receive_beamwidth"]) / 10.0 * math.pi / 180
    # DUVIDA DIOGO
    TVG_xover = 1.0 / np.cos(math.pi / 180 * float(run_dgs[0]["TVG_law_crossover_ang"]))
    # TVG_xover = float(run_dgs[0]['TVG_law_crossover_ang'])*math.pi/180
    fac = 1.0 / np.sqrt(TVG_xover - 1.0)

    wtfall_img = no_val * np.ones([len(sid_dgs), W])  # , np.int16)
    angles_img = no_val * np.ones([len(sid_dgs), W])  # , np.int16)
    reflec_img = no_val * np.ones([len(sid_dgs), W])  # , np.int16)
    for n, sid_dg in enumerate(sid_dgs):
        (
            smps,
            xs,
            ys,
            zs,
            travel_times,
            cnt_smps,
            reflectivity,
            Ns,
            smp_freq,
            sound_speed,
            ob_bs,
            ni_bs,
            Ro,
        ) = get_waterfall_info(sid_dg, xyz_dgs[n], rra_dgs[n])
        # Get pixel positions from Y values of central samples in beams
        waterfall_ys = np.round(ys / pixel_size + max_y).astype(int)
        # Get a range from min() to max() of pixel positions
        valid_interval = range(np.amin(waterfall_ys), np.amax(waterfall_ys) + 1)
        # Find travel times for this range by linear interpolation
        # of travel times as a function of pixel positions of central samples
        full_travel_times = np.interp(valid_interval, waterfall_ys, travel_times)
        # Find travel times' nearest neighbours 'cnt_travel_times'
        # to subtract from 'full_travel_times'
        f = interpolate.interp1d(waterfall_ys, travel_times, kind="nearest")
        cnt_travel_times = f(valid_interval)
        # Find nearest neighbours of 'cnt_smps'
        f = interpolate.interp1d(waterfall_ys, cnt_smps, kind="nearest")
        full_cnt_smps = f(valid_interval)
        # Go to sample numbers in beams
        sample_ns = np.subtract(full_travel_times, cnt_travel_times) * smp_freq
        sample_ns = np.add(sample_ns, full_cnt_smps)
        sample_ns = np.clip(sample_ns.astype(int), 0, len(smps) - 1)
        # Copy beams to waterfall
        wtfall_img[n, valid_interval] = smps[sample_ns]  # np.absolute(smps[sample_ns])

        # Calculate measured return angles for vectors xs, ys and zs
        cur_lengths = np.sqrt((xs**2) + (ys**2))
        cur_lengths[ys < 0] *= -1
        cur_measured_angles = np.arctan2(cur_lengths, zs) * 180 / math.pi
        angles_img[n, valid_interval] = np.round(
            np.interp(valid_interval, waterfall_ys, cur_measured_angles)
        )

        # Calculate waterfall correction to create
        # the non-corrected waterfall
        sinangle = np.absolute(
            np.sin(np.absolute(angles_img[n, valid_interval]) * math.pi / 180)
        )
        sinangle[sinangle < 0.01] = 0.01
        cosangle = np.absolute(
            np.cos(np.absolute(angles_img[n, valid_interval]) * math.pi / 180)
        )
        cosangle[cosangle < 0.01] = 0.01
        range_resolution = sound_speed * pulse_width / 2.0
        xyz_hypothenuse = np.interp(
            valid_interval, waterfall_ys, np.sqrt((xs**2) + (ys**2) + (zs**2))
        )
        aTrackBeam_meters = xyz_hypothenuse * transmit_beamwidth
        rx_transducer_length = 2.1
        xTrackBeam_meters = rx_transducer_length
        area1 = xTrackBeam_meters * aTrackBeam_meters / cosangle
        area2 = range_resolution / sinangle * aTrackBeam_meters / cosangle
        area_m2 = np.minimum(area1, area2)
        area_focused = np.zeros(area_m2.shape)
        cur_vals = area_m2 > 0
        area_focused[cur_vals] = -10.0 * np.log10(area_m2[cur_vals])
        xTrackBeam_meters = xyz_hypothenuse * receive_beamwidth
        area1 = xTrackBeam_meters * aTrackBeam_meters / cosangle
        area2 = range_resolution * aTrackBeam_meters / sinangle
        area_m2 = np.minimum(area1, area2)
        area_simrad = np.zeros(area_m2.shape)
        cur_vals = area_m2 > 0
        area_simrad[cur_vals] = -10.0 * np.log10(area_m2[cur_vals])
        area = area_focused - area_simrad
        boresight_time = (full_travel_times * 2.0) * smp_freq
        # lambert = -20.0 * np.log10(cosangle)
        lambert = 20.0 * np.log10(boresight_time / Ro)
        dbs = np.zeros(area_m2.shape)
        dbs[boresight_time <= Ro] = ob_bs - ni_bs
        cur_vals = (boresight_time > Ro) & (boresight_time <= TVG_xover * Ro)
        dbs[cur_vals] = (1.0 - fac * np.sqrt(boresight_time[cur_vals] / Ro - 1.0)) * (
            ob_bs - ni_bs
        )
        # Calculate real reflectivity of samples
        correction = area - dbs - lambert  # *10.0 # Samples are in 0.1dB*10
        reflec_img[n, valid_interval] = wtfall_img[n, valid_interval] + correction
    wtfall_img = waterfall_correction_by_angle(wtfall_img, angles_img, no_val)
    # Calculate average center frequency and average sound speed
    centre_freq = np.array(
        [np.array(rra_dg["centre_freq"]).mean() for rra_dg in rra_dgs]
    ).mean()
    sound_speed = np.array([rra_dg["soundspeed"] for rra_dg in rra_dgs]).mean() / 10
    return wtfall_img, angles_img, reflec_img, no_val, centre_freq, sound_speed


### Backscatter waterfall normalization by angle
def waterfall_correction_by_angle(wtfall_img, angles_img, no_val):
    correction_mode = 0
    if correction_mode == 0:
        window_size = 601
        angl_ref = 45
        wtf_nok = wtfall_img == no_val
        H = wtfall_img.shape[0]
        W = wtfall_img.shape[1]
        un_angls = np.unique(angles_img)
        un_angls = un_angls[un_angls != no_val]
        angl_ref_i = np.where(un_angls == angl_ref)[0][0]
        avgs_by_angle = np.zeros((H, len(un_angls)))
        for i in range(H):
            # for k,an in enumerate(un_angls):
            avgs_by_angle[i, :] = [
                np.mean(wtfall_img[i, angles_img[i, :] == an]) for an in un_angls
            ]
            avgs_by_angle[i, np.argwhere(np.isnan(avgs_by_angle[i, :]))] = 0
        avgs_by_angle = signal.convolve2d(
            avgs_by_angle, np.ones((window_size, 1)) / window_size, "same"
        )
        out_wtfall_img = np.copy(wtfall_img)
        for i in range(H):
            avgs_by_angle[i, :] -= avgs_by_angle[i, angl_ref_i]
            for k, an in enumerate(un_angls):
                cur_ok = np.where(angles_img[i, :] == an)
                out_wtfall_img[i, cur_ok] -= avgs_by_angle[i, k]
        out_wtfall_img[wtf_nok] = no_val
    elif correction_mode == 1:
        wtf_ok = wtfall_img != no_val
        org_wtf_min = wtfall_img[wtf_ok].min()
        for k in np.unique(angles_img):
            if k == no_val:
                continue
            cur_ok = np.where(angles_img == k)
            wtfall_img[cur_ok] -= np.average(wtfall_img[cur_ok])
        new_wtf_min = wtfall_img[wtf_ok].min()
        wtfall_img[wtf_ok] -= new_wtf_min - org_wtf_min
        out_wtfall_img = wtfall_img
    return out_wtfall_img


def get_ground_control_points(pos_dgs, xyz_dgs, wtfall_img, pixel_size, bad_val):
    t_pos = get_datenum(pos_dgs)
    t_xyz = get_datenum(xyz_dgs)

    # xs, ys, epsg_code = geoutm(pos_dgs)
    latitude = np.asarray([p["latitude"] for p in pos_dgs])
    longitude = np.asarray([p["longitude"] for p in pos_dgs])
    xs, ys, epsg_code = geoutm(latitude, longitude)

    if t_pos != sorted(t_pos):
        t_pos_i = np.argsort(t_pos)
        t_pos = sorted(t_pos)
        xs = [xs[i] for i in t_pos_i]
        ys = [ys[i] for i in t_pos_i]
    heading = (
        np.asarray([float(xyz["heading_dot01"]) for xyz in xyz_dgs])
        * math.pi
        / 180
        / 100
    )

    #t_pos   = np.array(t_pos)
    #t_xyz   = np.array(t_xyz)
    #xyz_oks = (t_xyz>=min(t_pos))&(t_xyz<=max(t_pos))
    #t_xyz   = t_xyz[xyz_oks]
    #heading = heading[xyz_oks]
    fx = interpolate.splrep(t_pos, xs)
    fy = interpolate.splrep(t_pos, ys)
    xs_utm_xyz = interpolate.splev(t_xyz, fx)
    ys_utm_xyz = interpolate.splev(t_xyz, fy)

    hN = 51
    heading_pad = np.pad(heading, (hN, hN), "edge")
    heading_pad = signal.fftconvolve(
        heading_pad, np.ones(hN) / hN, mode="same"
    )  ### Low-pass filtering for heading values
    # heading_pad = signal.convolve(heading_pad, np.ones(hN)/hN, mode='same', method='fft') ### Low-pass filtering for heading values
    heading = heading_pad[hN:(-hN)]

    Ny = 30
    Nx = 12
    H = wtfall_img.shape[0]
    W = wtfall_img.shape[1]
    center_x = (W - 1) / 2
    ys = np.round(np.linspace(0, H - 1, Ny)).astype(int)
    # Avoid value extrapolation
    xyz_noks = (np.array(t_xyz)<min(t_pos))|(np.array(t_xyz)>max(t_pos))
    ok_xyzs = 0
    for y in ys:
        if xyz_noks[y]:
            continue
        cur_geo_xy = np.zeros((Nx, 2))
        cur_img_xy = np.zeros((Nx, 2))
        good_vals = np.nonzero(wtfall_img[y, :] != bad_val)[0]
        xs = np.round(np.linspace(0, len(good_vals) - 1, Nx)).astype(int)
        for xi, x in enumerate(good_vals[xs]):
            dx = (x - center_x) * pixel_size * math.cos(heading[y])
            dy = (x - center_x) * pixel_size * math.sin(heading[y])
            cur_geo_xy[xi, :] = [float(xs_utm_xyz[y] + dx), float(ys_utm_xyz[y] - dy)]
            cur_img_xy[xi, :] = [float(x), float(y)]
        # TO-DO: trabalhar com https://towardsdatascience.com/the-concave-hull-c649795c0f0f
        if ok_xyzs>2: #y > ys[1]:
            ok_pts = outside_hull(cur_geo_xy, geo_xy)
            geo_xy = np.append(geo_xy, cur_geo_xy[ok_pts, :], axis=0)
            img_xy = np.append(img_xy, cur_img_xy[ok_pts, :], axis=0)
        elif ok_xyzs>1: #y >= ys[1]:
            geo_xy = np.append(geo_xy, cur_geo_xy, axis=0)
            img_xy = np.append(img_xy, cur_img_xy, axis=0)
        else:
            geo_xy = np.copy(cur_geo_xy)
            img_xy = np.copy(cur_img_xy)
        ok_xyzs += 1
    gcpList = []
    for k in range(0, len(geo_xy)):
        gcpList.append(
            gdal.GCP(geo_xy[k][0], geo_xy[k][1], 0, img_xy[k][0], img_xy[k][1])
        )
    # show_pts(geo_xy[:,0], geo_xy[:,1], 'Georref.')
    # show_pts(img_xy[:,0], img_xy[:,1], 'Img positions')
    return gcpList, epsg_code
    # for y in range(H):
    #     good_vals = np.nonzero(wtfall_img[y,:]!=bad_val)[0]
    #     cur_geo_xy = np.zeros((len(good_vals),2))
    #     for xi, x in enumerate(good_vals):
    #         dx = (x-center_x)*pixel_size*math.cos(heading[y])
    #         dy = (x-center_x)*pixel_size*math.sin(heading[y])
    #         cur_geo_xy[xi,:] = [float(xs_utm_xyz[y]+dx), float(ys_utm_xyz[y]-dy)]
    #     if y>0:
    #         geo_xy = np.append(geo_xy, cur_geo_xy, axis=0)
    #     else:
    #         geo_xy = np.copy(cur_geo_xy)
    # return gcpList, epsg_code, geo_xy


# def project_waterfall(x, y, s, pixel_size):
#     minx = x.min()
#     maxx = x.max()
#     miny = y.min()
#     maxy = y.max()
#     xi = np.linspace(minx,maxx,round((maxx-minx)/pixel_size)+1)
#     yi = np.linspace(miny,maxy,round((maxy-miny)/pixel_size)+1)
#     x_grid, y_grid = np.meshgrid(xi, yi)
#     gcpList = []
#     geo_xy = [[minx,miny],[minx,maxy],[maxx,maxy],[maxx,miny]]
#     img_xy = [[0,0],[0,x_grid.shape[1]-1],[x_grid.shape[0]-1,x_grid.shape[1]-1],[x_grid.shape[0]-1,0]]
#     for k in range(0,len(geo_xy)):
#         gcpList.append(gdal.GCP(geo_xy[k][0], geo_xy[k][1], 0, img_xy[k][0], img_xy[k][1]))
#     rbf_fun = Rbf(x, y, s)
#     return rbf_fun(x_grid.ravel(), y_grid.ravel()).reshape(x_grid.shape), gcpList

# self.update_textbox(file_name + '\n\tProjecting backscatter waterfall', "TIME_DIFF")
# geo_img, geo_gcps = project_waterfall(geo_xy[:,0], geo_xy[:,1], wtfall_img[wtfall_img!=self.new_bad_val], pixel_size)
# gc.collect()
# self.update_textbox(file_name + '\n\tSaving projected backscatter waterfall', "TIME_DIFF")
# gdal.Translate(
#     destName="%s_back_handmade_geo.tif" % f,
#     srcDS=geo_img,
#     GCPs=geo_gcps,
#     outputSRS = cur_epsg_code,
#     noData=None)
# gc.collect()


def image_normalization(img, bad_val, new_bad_val, min_val, max_val, alpha, beta):
    img_ok = img != bad_val
    img[img_ok] = np.clip(img[img_ok], min_val, max_val) * alpha + beta
    img[np.logical_not(img_ok)] = new_bad_val
    return img


def get_datenum(dgs):
    cur_seconds = [
        float(
            date.toordinal(
                date(
                    int(dg["date"] / 10000),
                    int((dg["date"] % 10000) / 100),
                    int(dg["date"] % 100),
                )
            )
        )
        * 24
        * 60
        * 60
        + float(dg["time"]) / 1000
        for dg in dgs
    ]
    return cur_seconds


def getUTMzone(lat, lng):
    # exceptions around Norway
    if (lat >= 56 and lat < 64) and (lng >= 3 and lng < 12):
        return 32
    # exceptions around Svalbard
    if lat >= 72 and lat < 84:
        if lng >= 0 and lng < 9:
            return 31
        if lng >= 9 and lng < 21:
            return 33
        if lng >= 21 and lng < 33:
            return 35
        if lng >= 33 and lng < 42:
            return 37
    return np.floor((lng + 180) / 6) + 1


# def geoutm(pos_dgs):
#     latitude  = np.asarray([p['latitude' ] for p in pos_dgs])
#     longitude = np.asarray([p['longitude'] for p in pos_dgs])
def geoutm(latitude, longitude):
    avg_lat = np.average(latitude)
    avg_lon = np.average(longitude)

    WGS84 = 1
    WGS84_ECCEN = 0.0818191908417579
    WGS84_RADIUS = 6378137  #  units are meters
    GRS80_FLATT = 298.257223563  # which is NAD83
    WGS84_FLATT = 298.257222101
    # h'mmm funnily enough these are reversed in the Strang and Borre book...
    NAD27 = 2
    NAD27_ECCEN = 0.082271854
    NAD27_RADIUS = 6378206.4  # units are meters
    CLK66 = 3
    CLK66_ECCEN = 0.082271684
    CLK66_RADIUS = 6378206.4

    k0 = 0.9996  # central scale factor
    x0 = 500000.0  # x0 to be added in standard UTM
    y0 = 10000000.0  # y0 to be added in standard UTM for southern hemisphere

    ellipsoid = 1
    e2 = WGS84_ECCEN**2
    a = WGS84_RADIUS
    # switch ellipsoid
    #     case WGS84
    #         e2 = WGS84_ECCEN^2;
    #         a = WGS84_RADIUS;
    #     case NAD27
    #         e2 = NAD27_ECCEN^2;
    #         a = NAD27_RADIUS;
    #     case CLK66
    #         e2 = CLK66_ECCEN^2;
    #         a = CLK66_RADIUS;
    # end

    e4 = e2**2
    e6 = e2**3
    m1 = 1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0
    m2 = 3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0
    m3 = 15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0
    m4 = 35.0 * e6 / 3072.0
    epr2 = e2 / (1.0 - e2)

    dphi = latitude * math.pi / 180  # dphi = phi*math.pi/180
    dlam = longitude * math.pi / 180  # dlam = lam*math.pi/180

    if avg_lat < 0:
        epsg_code = "EPSG:327%d" % getUTMzone(avg_lat, avg_lon)
    else:
        epsg_code = "EPSG:326%d" % getUTMzone(avg_lat, avg_lon)

    utm_longs = np.asarray(range(-180, 181))
    utm_cnt_mer = np.floor(utm_longs / 6) * 6 + 3
    clam = utm_cnt_mer[utm_longs == np.round(avg_lon)]  # -57
    dclam = clam * math.pi / 180
    cosphi = np.asarray([math.cos(dp) for dp in dphi])
    sinphi = np.asarray([math.sin(dp) for dp in dphi])
    tanphi = np.asarray([math.tan(dp) for dp in dphi])
    n = np.asarray([a / math.sqrt(1.0 - e2 * (sp**2)) for sp in sinphi])
    t = tanphi**2
    t2 = t**2
    c = epr2 * (cosphi**2)
    b = (dlam - dclam) * cosphi  # b is A in Snyder's formulas
    b2 = b * b
    b3 = b2 * b
    b4 = b3 * b
    b5 = b4 * b
    b6 = b5 * b
    dummy1 = np.asarray([math.sin(2.0 * dp) for dp in dphi])
    dummy2 = np.asarray([math.sin(4.0 * dp) for dp in dphi])
    dummy3 = np.asarray([math.sin(6.0 * dp) for dp in dphi])
    m = a * (m1 * dphi - m2 * dummy1 + m3 * dummy2 - m4 * dummy3)
    xs = (
        k0
        * n
        * (
            b
            + (1.0 - t + c) * b3 / 6.0
            + (5.0 - 18.0 * t + t2 + 72.0 * c - 58.0 * epr2) * b5 / 120.0
        )
    )
    xs = xs + x0
    ys = k0 * (
        m
        + n
        * tanphi
        * (
            b2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * (c**2)) * b4 / 24.0
            + (61.0 - 58.0 * t + t2 + 600.0 * c - 330.0 * epr2) * b6 / 720.0
        )
    )
    if max(longitude) < 0.0:  # (hemisphere ~= 0):
        ys = ys + y0
    return xs, ys, epsg_code


def notch_filtering(img_array, bad_val):
    #  HUNG, EDSON MINTSU ; NETO, ARTHUR AYRES ; MAGRANI, FÁBIO JOSÉ GUEDES . Waterfall notch-filtering for restoration of acoustic backscatter records from Admiralty Bay, Antarctica. MARINE GEOPHYSICAL RESEARCH
    H = img_array.shape[0]
    W = img_array.shape[1]
    theta = 1.0
    delta = 0.025
    G = 10 ** (-40 / 10)
    T = np.fft.fftshift(np.fft.fft2(img_array.astype(np.float64)))
    x, y = np.meshgrid(np.linspace(-0.5, 0.5, W), np.linspace(-0.5, 0.5, H))
    notch_mask = np.absolute(np.arctan(x / y)) <= ((theta / 2) * math.pi / 180)
    notch_mask[np.absolute(y) <= delta] = 0
    T[notch_mask] *= G
    t = np.real(np.fft.ifft2(np.fft.ifftshift(T))) + 0.5
    t = t.astype(img_array.dtype)
    t[img_array == bad_val] = bad_val
    return t


def force_remove(cur_remove_file):
    cur_remove_folder_name, cur_remove_file_name = os.path.split(cur_remove_file)
    if cur_remove_file_name in os.listdir(cur_remove_folder_name):
        os.remove(cur_remove_file)


def save_numpy_array(img_array, file_name):
    if img_array.dtype == "uint16":
        data_type = gdal.GDT_UInt16
    elif img_array.dtype == "int16":
        data_type = gdal.GDT_Int16
    else:  # if img_array.dtype===='uint8':
        data_type = gdal.GDT_Byte
    # Create(file_name, width, height, num_bands, data_type)
    dst_ds = gdal.GetDriverByName("GTiff").Create(
        file_name, img_array.shape[1], img_array.shape[0], 1, data_type
    )
    dst_ds.GetRasterBand(1).WriteArray(img_array)


def translate_warp(input_tif, input_points, epsg_code, output_tif):
    # resampleAlg = gdal.GRIORA_NearestNeighbour
    resampleAlg = gdal.GRIORA_Bilinear
    # resampleAlg = gdal.GRA_Bilinear
    # transformerOptions = "SRC_METHOD=GCP_POLYNOMIAL"
    # transformerOptions=transformerOptions,
    # resampleAlg = gdal.GRIORA_CubicSpline
    # resampleAlg = gdal.GRIORA_Lanczos
    use_thinplate = True  # False
    errorThreshold = 200
    polynomialOrder = 3
    gtif = gdal.Translate(
        "",
        gdal.Open(input_tif),
        GCPs=input_points,
        outputSRS=epsg_code,
        noData=None,  # 0,
        format="MEM",
    )
    if not output_tif:
        return gdal.Warp(
            "",
            gtif,
            format="MEM",
            srcSRS=epsg_code,
            dstSRS=epsg_code,
            resampleAlg=resampleAlg,
            tps=use_thinplate,
            # polynomialOrder=polynomialOrder,
            errorThreshold=errorThreshold,
        )  # ,
        # dstAlpha=True,
        # srcNodata=0,
        # dstNodata=0)
    else:
        if os.path.isfile(output_tif):
            os.remove(output_tif)
        gdal.Warp(
            output_tif,
            gtif,
            format="GTiff",
            srcSRS=epsg_code,
            dstSRS=epsg_code,
            resampleAlg=resampleAlg,
            tps=use_thinplate,
            # polynomialOrder=polynomialOrder,
            errorThreshold=errorThreshold,
        )  # ,
        # dstAlpha=True,
        # srcNodata=0,
        # dstNodata=0)


def tiff_to_jp2(input_tif, output_jp2, epsg_code):
    gdal.Translate(
        destName=output_jp2, srcDS=input_tif, outputSRS=epsg_code, noData=None
    )
    os.remove(input_tif)


# def tiff_to_jp2(input_tif, output_jp2, quality_value, is_lossless):
#     if is_lossless:
#         gdal.GetDriverByName('JP2OpenJPEG').CreateCopy(
#             output_jp2, gdal.Open(input_tif), 0,
#             ['QUALITY=100','REVERSIBLE=YES'])
#     else:
#         gdal.GetDriverByName('JP2OpenJPEG').CreateCopy(
#             output_jp2, gdal.Open(input_tif), 0,
#             ['QUALITY=%d' % quality_value])
#     os.remove(input_tif)


def save_gcp_points(out_pts_file, input_points):
    with open(out_pts_file, "w") as f:
        f.write("mapX,mapY,pixelX,pixelY,enable\n")
        for gcp in input_points:
            f.write(
                "%f,%f,%d,-%d,1\n" % (gcp.GCPX, gcp.GCPY, gcp.GCPPixel, gcp.GCPLine)
            )


def translate_warp_tif(
    input_tif, input_points, epsg_code, output_tif, is_lossless, save_wtf
):
    resampleAlg = gdal.GRIORA_Bilinear
    # resampleAlg = gdal.GRIORA_NearestNeighbour
    # resampleAlg = gdal.GRA_Bilinear
    # resampleAlg = gdal.GRIORA_CubicSpline
    # resampleAlg = gdal.GRIORA_Lanczos
    # transformerOptions = "SRC_METHOD=GCP_POLYNOMIAL"
    use_thinplate = True
    errorThreshold = 200
    polynomialOrder = 3
    out_pts1 = input_tif + ".points"
    save_gcp_points(out_pts1, input_points)
    out_tif1 = input_tif + "_translate.tif"
    gdal.Translate(
        destName=out_tif1,
        srcDS=input_tif,  # gdal.Open(input_tif),
        GCPs=input_points,
        outputSRS=epsg_code,
        noData=None,
    )  # , format='MEM')
    gdal.Warp(
        destNameOrDestDS=output_tif,
        srcDSOrSrcDSTab=out_tif1,
        srcSRS=epsg_code,
        dstSRS=epsg_code,
        resampleAlg=resampleAlg,
        tps=use_thinplate,
        errorThreshold=errorThreshold,
    )  # , format='MEM')
    if not save_wtf:
        os.remove(input_tif)
        os.remove(out_pts1)
    os.remove(out_tif1)


def get_shape_cut(input_tif, input_shape, output_tif):
    epsg_code = QgsRasterLayer(input_tif).crs().authid()
    try:
        gdal.Warp(
            output_tif,
            input_tif,
            srcSRS=epsg_code,  # format="GTiff",
            dstSRS=epsg_code,
            dstAlpha=True,
            srcNodata=0,
            dstNodata=0,
            cutlineDSName=input_shape,
            cropToCutline=True,
        )
    except:
        return False
    return True


def load_image_QGIS(file_name, layer_name, min_val, max_val):
    rasterLyr = QgsRasterLayer(file_name, layer_name)
    myEnhancement = QgsContrastEnhancement(
        rasterLyr.renderer().dataType(rasterLyr.renderer().grayBand())
    )
    myEnhancement.setContrastEnhancementAlgorithm(
        QgsContrastEnhancement.StretchAndClipToMinimumMaximum,  # StretchToMinimumMaximum,
        True,
    )
    myEnhancement.setMinimumValue(min_val)
    myEnhancement.setMaximumValue(max_val)
    rasterLyr.renderer().setContrastEnhancement(myEnhancement)
    rasterLyr.isValid()
    QgsProject.instance().addMapLayers([rasterLyr])
    # QgsMapLayerRegistry.instance().addMapLayers([rasterLyr])


def read_img(img_name):
    a = gdal.Open(img_name, gdal.GA_ReadOnly)
    b = a.GetRasterBand(1)
    c = b.ReadAsArray()
    return np.asarray(c)


def floating_range(start, stop, steps):
    return [(stop - start) / (steps - 1) * a + start for a in range(steps)]


def Create_Multipoint_Shapefile(
    shapefile_name, xs, ys, epsg_code, layer_name, attrs, attr_names, cur_map_tip
):
    vect_lyr_call = "Point?crs=%s" % epsg_code
    for a in attr_names:
        vect_lyr_call += "&field=%s:double" % a
    vl = QgsVectorLayer(vect_lyr_call, layer_name, "memory")
    QgsProject.instance().addMapLayer(vl)
    pr = vl.dataProvider()
    for count in range(len(xs)):
        # add a feature
        fet = QgsFeature()
        fet.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(xs[count], ys[count])))
        fet.setAttributes(attrs[:, count].tolist())
        pr.addFeatures([fet])
    vl.updateExtents()
    # print(vl.renderer().symbol().symbolLayer(0).properties())
    vl = Graduate_Multipoint_Shapefile(vl, attrs[0, :], attr_names[0], cur_map_tip)
    QgsVectorFileWriter.writeAsVectorFormat(
        vl,
        shapefile_name,
        "utf-8",
        vl.crs(),
        "ESRI Shapefile",
        symbologyExport=QgsVectorFileWriter.SymbolLayerSymbology,
    )  # , layerOptions=["mapTipTemplate='" + cur_map_tip + "'"])
    return vl


def Graduate_Multipoint_Shapefile(shp_layer, attrs, attr_name, cur_map_tip):
    steps = 10
    x = floating_range(attrs.min(), attrs.max(), steps + 1)
    lower = x[0:-1]
    upper = [a + 0.001 for a in x[1:]]
    names = [
        "%s de %1.1f a %1.1f" % (attr_name, lower[a], upper[a]) for a in range(steps)
    ]
    ns = floating_range(64, 255, steps)
    colors = ["#FF%X%X" % (int(round(a)), int(round(a))) for a in ns]
    cur_ranges = []
    for n in range(steps):
        sym = QgsSymbol.defaultSymbol(shp_layer.geometryType())
        sym.setColor(QColor(colors[n]))
        p = sym.symbolLayer(0).properties()
        p["outline_style"] = "no"
        sym.changeSymbolLayer(0, QgsSimpleMarkerSymbolLayer.create(p))
        rng = QgsRendererRange(lower[n], upper[n], sym, names[n])
        cur_ranges.append(rng)

    shp_layer.setRenderer(QgsGraduatedSymbolRenderer(attr_name, cur_ranges))
    shp_layer.setMapTipTemplate(cur_map_tip)
    shp_layer.updateFields()
    shp_layer.updateExtents()
    shp_layer.commitChanges()
    return shp_layer


def calc_hist_by_angle(
    angle_file, refl_file, angl_prop, refl_prop, min_refl_val, use_absolute_ang
):
    angls = read_img(angle_file)
    refls = read_img(refl_file)
    # (-180.0,1.0), (-100.0,0.01), True
    refls = refls * refl_prop[1] + refl_prop[0]
    angls = angls * angl_prop[1] + angl_prop[0]
    angls = angls[refls >= min_refl_val]
    refls = refls[refls >= min_refl_val]
    if use_absolute_ang:
        angls = np.absolute(angls)
        hist_angls = np.asarray(range(91))
        hist_refls = np.zeros(91)
        hist_cnts = np.zeros(91)
        for cur_angl in range(91):
            cur_vals = angls == cur_angl
            hist_refls[cur_angl] = refls[cur_vals].sum()
            hist_cnts[cur_angl] = np.count_nonzero(cur_vals)
    else:
        hist_angls = np.asarray(range(-90, 91))
        hist_refls = np.zeros(181)
        hist_cnts = np.zeros(181)
        for cur_angl in range(181):
            cur_vals = angls == (cur_angl - 90)
            hist_refls[cur_angl] = refls[cur_vals].sum()
            hist_cnts[cur_angl] = np.count_nonzero(cur_vals)

    del angls
    del refls
    return hist_angls, hist_refls, hist_cnts


LIMIT_BLUNDER = 1.0
LIMIT_NEAR = 10.0
LIMIT_PROX = 20.0
LIMIT_FAR = 35.0
LIMIT_DIST = 55.0
LIMIT_OUTER = 65.0
LIMIT_AWAY = 85.0


class Invert_Model:
    def __init__(self, angls, refls, freq, nu_water):
        # function  [z_out,vol_out,rough_out]=AnnealingPetrobrasModel (freq_in,angle_in,AR_in)
        # # freq_in - frequencia de entrada em Hz
        # # angle_in  vetor linha com ângulos de incidência de 0 a 90 graus
        # # AR_in     vetor linha com backscatter em decibels
        self.angls = angls
        self.refls = refls
        self.freq = freq
        self.nu_water = nu_water

        self.k0 = 2.0 * math.pi * self.freq / self.nu_water

        # clipar dados nos ângulos de incidência de [0,60] graus e backscatter [-60,-4] dB
        self.angls = np.absolute(self.angls)
        valid_index = (
            (self.angls <= 60.0)
            & (self.angls >= 0)
            & (self.refls > -60.0)
            & (self.refls < -4.0)
        )
        self.angls = self.angls[valid_index]
        self.refls = self.refls[valid_index]
        self.update_theta()

        # limite dos parâmetros do modelo
        z_min = 1.05
        z_max = 5.0
        vol_min = 0.001
        vol_max = 100.0
        rough_max = 86400 / self.freq
        rough_min = rough_max / 100

        # Valores iniciais
        param = [z_min, vol_min, rough_min]

        # Otimização por Annealing
        j = self.custo(param)
        j_min = j
        param_min = param

        t = 0.1  # temperatura
        eps = 0.01  # perturbação
        N_INT = 10000  # número de iterações

        for k in range(10):
            t = t / np.log2(2.0 + float(k))
            for i in range(N_INT):
                # new_param = param + eps*rand(1,3,'normal')
                new_param = param + eps * np.random.randn(3)
                new_param = np.clip(
                    new_param, [z_min, vol_min, rough_min], [z_max, vol_max, rough_max]
                )
                new_j = self.custo(new_param)
                delta_j = j - new_j
                q = np.exp(delta_j / t)
                if np.random.rand() < q:
                    j = new_j
                    param = new_param
                if new_j < j_min:
                    j_min = new_j
                    param_min = new_param

        ### Parâmetros de saída
        # impedance = Impedance
        # sigma2 = volume
        # ch0 = rugosity
        self.impedance = param_min[0] * 1.022 * 1.5
        self.sigma2 = param_min[1]
        self.ch0 = param_min[2]

    def update_theta(self):
        teta = self.angls * math.pi / 180.0
        teta[np.absolute(teta) < 1.0e-5] = 1.0e-5
        self.teta_sin_2 = np.sin(teta) ** 2
        self.teta_cos_2 = np.cos(teta) ** 2
        self.teta_cos_4 = self.teta_cos_2**2

    def petrobras_model(self, param):
        # nu_water = 1500.0
        R90 = (param[0] - 1.0) / (param[0] + 1.0)
        R90_2 = np.absolute(R90) ** 2
        T90_2 = (1.0 - R90_2) ** 2

        # Interface Model
        # delta = 2.0e-3 * param[2] * k
        delta = 2.0e-4 * (param[2] ** 2) * self.k0
        sigma_vv = (
            delta
            * R90_2
            / (8.0 * math.pi)
            / ((self.teta_sin_2 + (delta**2) * self.teta_cos_4) ** 1.5)
        )

        # Volume
        # sigma_v =  1.0e-6 * param[1] * k * T90_2  * (np.cos(teta)**2.0)
        sigma_v = 1.0e-5 * param[1] * (self.k0 ** (2 / 3)) * T90_2 * self.teta_cos_2

        self.S_total = 10.0 * np.log10(sigma_vv + sigma_v)
        self.S_volume = 10.0 * np.log10(sigma_v)
        self.S_interface = 10.0 * np.log10(sigma_vv)

    def custo(self, param):
        self.petrobras_model(param)
        return np.sqrt(np.sum((self.refls - self.S_total) ** 2))

    def plotar_resultados(self, file_name):
        matplotlib.interactive(True)
        plt.figure()
        plt.plot(self.angls, self.refls, "b")
        plt.plot(-self.angls, self.refls, "b")
        angls_dummy = self.angls
        S_total_dummy = self.S_total
        S_volume_dummy = self.S_volume
        S_interface_dummy = self.S_interface
        self.angls = np.array([float(n) / 4 for n in range(241)])  # 0:0.25:60
        self.update_theta()
        self.petrobras_model([self.z_out / 1.022 / 1.5, self.vol_out, self.rough_out])
        plt.plot(self.angls, self.S_total, "r")
        plt.plot(-np.array(self.angls), self.S_total, "r")
        plt.title(
            "%s\nfreq=%2.0fkHz, z=%1.2f, v=%1.2f, r=%1.2f"
            % (file_name, self.freq / 1000, self.z_out, self.vol_out, self.rough_out)
        )
        plt.axis([-65, 65, -40, 0])
        plt.show()
        self.angls = angls_dummy
        self.update_theta()
        self.S_total = S_total_dummy
        self.S_volume = S_volume_dummy
        self.S_interface = S_interface_dummy


# Weibull distribution fitting for histogram Hin
class RedeNeuralWeibull6:
    def __init__(self, Hin):
        Hin = np.array(Hin)
        self.Hin = Hin
        self.N = Hin.size
        self.x = np.arange(self.N) / self.N  # [0:N-1]/N
        self.mean_Hin = np.mean(self.Hin)
        index_max = np.argmax(self.Hin)
        self.max_Hin = self.Hin[index_max]
        delta = 0.0001
        mu = 0.2
        # self.a = x[index_max] / (((self.b-1)/self.b)**(1/self.b))
        a = self.x[index_max] * 1.2
        b = 2.0
        erro = self.CalculaErro(a, b)
        k = 1000
        for i in range(k):
            da, db = self.CalculaDerivadas(a, b, delta)
            a = a - da * mu / 50
            a = np.clip(a, 0.005, None)
            b = b - db * mu
            erro_novo = self.CalculaErro(a, b)
            if erro_novo > erro:
                mu = mu / 2.0
            if (np.absolute(db) < 1e-5) & (np.absolute(da) < 1e-3):
                break
            erro = erro_novo
        self.a = a  # lambda
        self.b = b  # k
        self.pdf = self.WeibullDistribution(a, b)

    def WeibullDistribution(self, a, b):
        a = np.clip(a, 0.005, 1.00)
        b = np.clip(b, 1.2, None)
        pdf = ((self.x / a) ** (b - 1)) * (b / a) * np.exp(-((self.x / a) ** b))
        pdf = pdf * self.mean_Hin / np.mean(pdf)
        return pdf

    def CalculaErro(self, a, b):
        pdf2 = self.WeibullDistribution(a, b)
        dif = pdf2 - self.Hin
        er = np.sum(dif**2) / (self.max_Hin**2)
        er = np.sqrt(er / (self.N + 1))
        return er

    def CalculaDerivadas(self, a, b, delta):
        erro_ref = self.CalculaErro(a, b)
        da = (self.CalculaErro(a + delta, b) - erro_ref) / delta
        db = (self.CalculaErro(a, b + delta) - erro_ref) / delta
        return da, db

    def plot_pdf(self, fig_name):
        matplotlib.interactive(True)
        plt.figure()
        plt.subplot(111)
        plt.plot(self.x, self.Hin, "b")
        plt.plot(self.x, self.pdf, "r")
        plt.grid(True)
        plt.show()
        plt.savefig(fig_name)
        plt.close()


#################################################
# FUNÇÕES DO DATA BINNING
#################################################

#################################################
# Leitura dos datagramas
#################################################
def read_pos_dep_datagrams_db(file_name):
    file_size = os.path.getsize(file_name)
    pos_dgs = []
    depth_dgs = []
    model_numbers = [
        1002,
        120,
        300,
        710,
        2000,
        3000,
        3020,
        302,
        122,
        121,
        850,
        2040,
        2045,
    ]
    endian = ">"  # '>' = big-endian, '<' = little-endian no módulo struct
    with open(file_name, "rb") as f:
        model = struct.unpack(">IBBH", f.read(8))[3]
    if model not in model_numbers:
        with open(file_name, "rb") as f:
            model = struct.unpack("<IBBH", f.read(8))[3]
        if model in model_numbers:
            endian = "<"
        else:
            raise ValueError(
                "Erro de modelo para o arquivo {}".format(os.path.split(file_name)[1])
            )

    with open(file_name, "rb") as f:
        while f.tell() < file_size:
            cur_dg_len = struct.unpack(endian + "I", f.read(4))[0]
            cur_dg_id = struct.unpack(endian + "cc", f.read(2))[1]
            cur_dg = f.read(cur_dg_len - 2)
            if checksum_error_db(cur_dg, cur_dg_id, endian):
                raise ValueError(
                    "Erro de checksum para o datagrama {}".format(cur_dg_id)
                )
            if cur_dg_id == b"P":
                pos_dgs.append(read_position_datagrams_db(cur_dg, endian))
            elif cur_dg_id == b"D":
                depth_dgs.append(read_depth_datagrams_db(cur_dg, endian))
            elif cur_dg_id == b"X":
                depth_dgs.append(read_xyz_datagrams_db(cur_dg, endian))
    return pos_dgs, depth_dgs, model


def read_seabed_datagrams_db(file_name):
    file_size = os.path.getsize(file_name)
    sid_dgs = []
    model_numbers = [
        1002,
        120,
        300,
        710,
        2000,
        3000,
        3020,
        302,
        122,
        121,
        850,
        2040,
        2045,
    ]
    endian = ">"  # '>' = big-endian, '<' = little-endian no módulo struct
    with open(file_name, "rb") as f:
        model = struct.unpack(">IBBH", f.read(8))[3]
    if model not in model_numbers:
        with open(file_name, "rb") as f:
            model = struct.unpack("<IBBH", f.read(8))[3]
        if model in model_numbers:
            endian = "<"
        else:
            raise ValueError(
                "Erro de modelo para o arquivo {}".format(os.path.split(file_name)[1])
            )

    with open(file_name, "rb") as f:
        while f.tell() < file_size:
            cur_dg_len = struct.unpack(endian + "I", f.read(4))[0]
            cur_dg_id = struct.unpack(endian + "cc", f.read(2))[1]
            cur_dg = f.read(cur_dg_len - 2)
            if checksum_error_db(cur_dg, cur_dg_id, endian):
                raise ValueError(
                    "Erro de checksum para o datagrama {}".format(cur_dg_id)
                )
            elif cur_dg_id == b"S":
                sid_dgs.append(read_sid_datagrams_db(cur_dg, endian))
            elif cur_dg_id == b"Y":
                sid_dgs.append(read_sid89_datagrams_db(cur_dg, endian))
    return sid_dgs


def checksum_error_db(cur_bytes, cur_dg_id, endian):
    dg_checksum = struct.unpack(endian + "H", cur_bytes[-2:])[0]
    cur_bytes = cur_bytes[:-3]
    cur_bytes_sum = sum(struct.unpack(endian + "B" * len(cur_bytes), cur_bytes))
    cur_bytes_sum += struct.unpack(endian + "B", cur_dg_id)[0]
    return (cur_bytes_sum & 65535) != dg_checksum


def read_initial_datagram_bytes_db(cur_bytes, field_names, field_types, endian):
    field_types = endian + field_types
    cur_index = struct.calcsize(field_types)
    field_values = struct.unpack(field_types, cur_bytes[:cur_index])
    cur_dg = dict(zip(field_names, field_values))
    return cur_dg, cur_index


def read_cycled_datagram_bytes_db(
    cur_bytes, cycle_len, field_names, field_types, endian
):
    field_types = endian + field_types * cycle_len
    cur_index = struct.calcsize(field_types)
    field_values = struct.unpack(field_types, cur_bytes[:cur_index])
    cur_dg = {}
    N = len(field_names)
    for k in range(N):
        cur_dg[field_names[k]] = field_values[k::N]
    return cur_dg, cur_index


def read_datagram_array_db(cur_bytes, array_len, field_name, field_type, endian):
    field_type = endian + field_type * array_len
    cur_index = struct.calcsize(field_type)
    cur_dg = {}
    cur_dg[field_name] = struct.unpack(field_type, cur_bytes[:cur_index])
    return cur_dg, cur_index


def read_position_datagrams_db(cur_bytes, endian):
    field_names = (
        "em_model",
        "date",
        "time",
        "pos_counter",
        "system_sn",
        "latitude",
        "longitude",
        "measure_fix_qual_cm",
        "speed_cm_s",
        "course_dot01",
        "heading_dot01",
        "pos_system_descriptor",
        "pos_in_dg_N",
    )
    field_types = "HIIHHiiHHHHBB"
    cur_dg = read_initial_datagram_bytes_db(
        cur_bytes, field_names, field_types, endian
    )[0]
    cur_dg["latitude"] = float(cur_dg["latitude"]) / 20000000
    cur_dg["longitude"] = float(cur_dg["longitude"]) / 10000000
    return cur_dg


def read_depth_datagrams_db(cur_bytes, endian):
    field_names = (
        "em_model",
        "date",
        "time",
        "ping",
        "system_sn",
        "heading_dot01",
        "sound_speed_dm_s",
        "depth_re_water_level",
        "N_max",
        "N",
        "z_res",
        "x_y_res",
        "smp_freq",
    )
    field_types = "HIIHHHHHBBBBH"
    cur_dg, cur_index = read_initial_datagram_bytes_db(
        cur_bytes, field_names, field_types, endian
    )
    field_names = (
        "z",
        "y",
        "x",
        "beam_dep_angle_dot01",
        "beam_azi_angle_dot01",
        "range_1way",
        "quality_factor",
        "length_det_win",
        "reflectivity_dot5dB",
        "beam_number",
    )
    field_types = "hhhhHHBBbB"
    cur_extra_dg, cur_index = read_cycled_datagram_bytes_db(
        cur_bytes[cur_index:], cur_dg["N"], field_names, field_types, endian
    )
    cur_dg.update(cur_extra_dg)
    cur_dg["z"] = [float(z) * float(cur_dg["z_res"]) * 0.01 for z in cur_dg["z"]]
    cur_dg["y"] = [float(y) * float(cur_dg["x_y_res"]) * 0.01 for y in cur_dg["y"]]
    cur_dg["x"] = [float(x) * float(cur_dg["x_y_res"]) * 0.01 for x in cur_dg["x"]]
    cur_dg["reflectivity"] = [float(bs) * 0.5 for bs in cur_dg["reflectivity_dot5dB"]]
    cur_dg["sound_speed"] = float(cur_dg["sound_speed_dm_s"]) * 0.1
    return cur_dg


def read_xyz_datagrams_db(cur_bytes, endian):
    field_names = (
        "em_model",
        "date",
        "time",
        "ping",
        "system_sn",
        "heading_dot01",
        "sound_speed_dm_s",
        "depth_re_water_level",
        "N",
        "N_valid",
        "smp_freq",
        "info",
        "spare1",
        "spare2",
        "spare3",
    )
    field_types = "HIIHHHHfHHfB" + 3 * "B"
    cur_dg, cur_index = read_initial_datagram_bytes_db(
        cur_bytes, field_names, field_types, endian
    )
    field_names = (
        "z",
        "y",
        "x",
        "length_det_win",
        "quality_factor",
        "beam_angle_adj_dot01",
        "detect_info",
        "real_time_info",
        "reflectivity_dot1dB",
    )
    field_types = "fffHBbBbh"
    cur_extra_dg, cur_index = read_cycled_datagram_bytes_db(
        cur_bytes[cur_index:], cur_dg["N"], field_names, field_types, endian
    )
    cur_dg.update(cur_extra_dg)
    cur_dg["reflectivity"] = [float(bs) * 0.1 for bs in cur_dg["reflectivity_dot1dB"]]
    cur_dg["sound_speed"] = float(cur_dg["sound_speed_dm_s"]) * 0.1
    return cur_dg


def read_sid_datagrams_db(cur_bytes, endian):
    field_names = (
        "em_model",
        "date",
        "time",
        "ping",
        "system_sn",
        "mean_abs_coef_db_km",
        "pulse_length_us",
        "range_norm_inc",
        "start_range_tvg",
        "stop_range_tvg",
        "bsn",
        "bso",
        "tx_beamwidth",
        "tvg_law",
        "N",
    )
    field_types = "HIIHHHHHHHbbHBB"
    cur_dg, cur_index1 = read_initial_datagram_bytes_db(
        cur_bytes, field_names, field_types, endian
    )
    field_names = ("beam_idx", "sort_dir", "Ns", "cnt_num")
    field_types = "BbHH"
    cur_extra_dg, cur_index2 = read_cycled_datagram_bytes_db(
        cur_bytes[cur_index1:], cur_dg["N"], field_names, field_types, endian
    )
    cur_dg.update(cur_extra_dg)
    cur_extra_dg = read_datagram_array_db(
        cur_bytes[(cur_index1 + cur_index2) :],
        sum(cur_dg["Ns"]),
        "samples",
        "b",
        endian,
    )[0]
    cur_dg.update(cur_extra_dg)
    cur_dg["samples"] = [float(sample) * 0.5 for sample in cur_dg["samples"]]
    return cur_dg


def read_sid89_datagrams_db(cur_bytes, endian):
    field_names = (
        "em_model",
        "date",
        "time",
        "ping",
        "system_sn",
        "smp_freq",
        "range_norm_inc",
        "bsn",
        "bso",
        "tx_beamwidth",
        "tvg_law",
        "N",
    )
    field_types = "HIIHHfHhhHHH"
    cur_dg, cur_index1 = read_initial_datagram_bytes_db(
        cur_bytes, field_names, field_types, endian
    )
    field_names = ("sort_dir", "detect_info", "Ns", "cnt_num")
    field_types = "BbHH"
    cur_extra_dg, cur_index2 = read_cycled_datagram_bytes_db(
        cur_bytes[cur_index1:], cur_dg["N"], field_names, field_types, endian
    )
    cur_dg.update(cur_extra_dg)
    cur_extra_dg = read_datagram_array_db(
        cur_bytes[(cur_index1 + cur_index2) :],
        sum(cur_dg["Ns"]),
        "samples",
        "h",
        endian,
    )[0]
    cur_dg.update(cur_extra_dg)
    cur_dg["bsn"] = float(cur_dg["bsn"]) * 0.1
    cur_dg["bso"] = float(cur_dg["bso"]) * 0.1
    cur_dg["samples"] = [float(sample) * 0.1 for sample in cur_dg["samples"]]
    return cur_dg


#################################################
# Interpolação e tratamento dos dados
#################################################


def db2amp(value: np.ndarray, no_val: float = 0) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        if no_val and value == no_val:
            return no_val
        else:
            return 10 ** (value / 20)
    arr = 10 ** (value / 20)
    arr[value == no_val] = no_val
    return arr


def amp2db(value: np.ndarray, no_val: float = 0) -> np.ndarray:
    if isinstance(value, float):
        if no_val and value == no_val:
            return no_val
        else:
            return 20 * np.log10(value)
    arr = 20 * np.log10(value)
    arr[value == no_val] = no_val
    return arr


def interpolate_data_db(position, depth):
    central_meridian, epsg_code = get_utm_db(position)

    for n in range(len(position)):
        pos_dgs = position[n]
        depth_dgs = depth[n]
        t_pos = get_datenum_db(pos_dgs)
        t_xyz = get_datenum_db(depth_dgs)
        for i, t in enumerate(t_pos):
            pos_dgs[i]["epoch"] = t
        for i, t in enumerate(t_xyz):
            depth_dgs[i]["epoch"] = t
        pos_dgs = organize_dtgs_db(pos_dgs)
        depth_dgs = organize_dtgs_db(depth_dgs)
        t_pos = [pos["epoch"] for pos in pos_dgs]
        t_xyz = [depth["epoch"] for depth in depth_dgs]

        latitude = np.asarray([p["latitude"] for p in pos_dgs])
        longitude = np.asarray([p["longitude"] for p in pos_dgs])
        xs, ys = geoutm_db(latitude, longitude, central_meridian)

        fx = interpolate.interp1d(t_pos, xs, fill_value="extrapolate")
        fy = interpolate.interp1d(t_pos, ys, fill_value="extrapolate")
        xs_utm_xyz = fx(t_xyz)
        ys_utm_xyz = fy(t_xyz)

        for i in range(len(xs_utm_xyz)):
            depth_dgs[i]["utmx"] = xs_utm_xyz[i]
            depth_dgs[i]["utmy"] = ys_utm_xyz[i]

        depth_dgs = locate_beams_db(depth_dgs)

        position[n] = pos_dgs
        depth[n] = depth_dgs

    return position, depth, epsg_code


def get_datenum_db(dgs):
    cur_seconds = [jday_db(dg["date"], dg["time"]) for dg in dgs]
    return cur_seconds


def jday_db(date, time):
    year = int(date / 10000)
    month = int((date % 10000) / 100)
    day = int(date % 100)
    year += 8000
    if month < 3:
        year -= 1
        month += 12
    jday = int(
        (year * 365)
        + (year / 4)
        - (year / 100)
        + (year / 400)
        - 1200820
        + (month * 153 + 3) / 5
        - 92
        + day
        - 1
    )
    jday -= 2440587  # Julian day since 1970
    return float(jday) * 24.0 * 60.0 * 60.0 + time / 1000.0


def organize_dtgs_db(dtgs):
    dtgs.sort(key=lambda x: x["epoch"])
    i = 1
    while i < len(dtgs):
        if dtgs[i]["epoch"] == dtgs[i - 1]["epoch"]:
            del dtgs[i]
        i += 1
    return dtgs


def sort_sid_db(sid_dgs):
    t_sid = get_datenum_db(sid_dgs)
    for i, t in enumerate(t_sid):
        sid_dgs[i]["epoch"] = t
    sid_dgs = organize_dtgs_db(sid_dgs)

    return sid_dgs


def get_soundspeed_db(depth_dgs):
    sound_speed = 0
    for depth_dg in depth_dgs:
        sound_speed += depth_dg["sound_speed"]
    sound_speed /= len(depth_dg)
    return sound_speed


def get_utm_db(position):
    latitude = []
    longitude = []
    for pos_dgs in position:
        lat = [pos["latitude"] for pos in pos_dgs]
        lon = [pos["longitude"] for pos in pos_dgs]
        latitude.extend(lat)
        longitude.extend(lon)
    min_lat = np.amin(latitude)
    max_lat = np.amax(latitude)
    mid_lat = max_lat + min_lat
    mid_lat = int(mid_lat / 2)
    min_lon = np.amin(longitude)
    max_lon = np.amax(longitude)
    mid_lon = max_lon + min_lon
    mid_lon = int(mid_lon / 2)

    utm_longs = np.asarray(range(-180, 181))
    utm_cnt_mer = np.floor(utm_longs / 6) * 6 + 3
    epsg_code = "EPSG:32"
    if mid_lat < 0:
        epsg_code += "7"
    else:
        epsg_code += "6"
    epsg_code = "{}{}".format(epsg_code, getUTMzone(mid_lat, mid_lon))
    central_meridian = utm_cnt_mer[utm_longs == np.round(mid_lon)]

    return central_meridian, epsg_code


def geoutm_db(latitude, longitude, central_meridian):
    # WGS84
    WGS84_ECCEN = 0.0818191908417579
    WGS84_RADIUS = 6378137  # units are meters

    k0 = 0.9996  # central scale factor
    x0 = 500000.0  # x0 to be added in standard UTM
    y0 = 10000000.0  # y0 to be added in standard UTM for southern hemisphere

    e2 = WGS84_ECCEN**2
    a = WGS84_RADIUS

    e4 = e2**2
    e6 = e2**3
    m1 = 1.0 - e2 / 4.0 - 3.0 * e4 / 64.0 - 5.0 * e6 / 256.0
    m2 = 3.0 * e2 / 8.0 + 3.0 * e4 / 32.0 + 45.0 * e6 / 1024.0
    m3 = 15.0 * e4 / 256.0 + 45.0 * e6 / 1024.0
    m4 = 35.0 * e6 / 3072.0
    epr2 = e2 / (1.0 - e2)

    dphi = latitude * math.pi / 180  # dphi = phi*math.pi/180
    dlam = longitude * math.pi / 180  # dlam = lam*math.pi/180

    clam = central_meridian  # -57
    dclam = clam * math.pi / 180
    cosphi = np.asarray([math.cos(dp) for dp in dphi])
    sinphi = np.asarray([math.sin(dp) for dp in dphi])
    tanphi = np.asarray([math.tan(dp) for dp in dphi])
    n = np.asarray([a / math.sqrt(1.0 - e2 * (sp**2)) for sp in sinphi])
    t = tanphi**2
    t2 = t**2
    c = epr2 * (cosphi**2)
    b = (dlam - dclam) * cosphi  # b is A in Snyder's formulas
    b2 = b * b
    b3 = b2 * b
    b4 = b3 * b
    b5 = b4 * b
    b6 = b5 * b
    dummy1 = np.asarray([math.sin(2.0 * dp) for dp in dphi])
    dummy2 = np.asarray([math.sin(4.0 * dp) for dp in dphi])
    dummy3 = np.asarray([math.sin(6.0 * dp) for dp in dphi])
    m = a * (m1 * dphi - m2 * dummy1 + m3 * dummy2 - m4 * dummy3)
    xs = (
        k0
        * n
        * (
            b
            + (1.0 - t + c) * b3 / 6.0
            + (5.0 - 18.0 * t + t2 + 72.0 * c - 58.0 * epr2) * b5 / 120.0
        )
    )
    xs = xs + x0
    ys = k0 * (
        m
        + n
        * tanphi
        * (
            b2 / 2.0
            + (5.0 - t + 9.0 * c + 4.0 * (c**2)) * b4 / 24.0
            + (61.0 - 58.0 * t + t2 + 600.0 * c - 330.0 * epr2) * b6 / 720.0
        )
    )
    if max(latitude) < 0.0:  # (hemisphere ~= 0):
        ys = ys + y0
    return xs, ys


def locate_beams_db(depth_dgs):
    for n, depth_dg in enumerate(depth_dgs):
        utmx = depth_dg["utmx"]
        utmy = depth_dg["utmy"]
        x = np.asarray(depth_dg["x"])
        y = np.asarray(depth_dg["y"])
        z = np.asarray(depth_dg["z"])
        length = np.sqrt((x**2) + (y**2))
        azimuth = depth_dg["heading_dot01"] * 0.01 * math.pi / 180 * np.ones(x.shape)
        azimuth[y > 0] += math.pi / 2 - np.arctan(x[y > 0] / np.abs(y[y > 0]))
        azimuth[y < 0] += -math.pi / 2 + np.arctan(x[y < 0] / np.abs(y[y < 0]))

        dy = length * np.cos(azimuth)
        dx = length * np.sin(azimuth)
        utmxs = utmx + dx
        utmys = utmy + dy

        full_length = np.sqrt((length**2) + (z**2))
        depression = np.arctan2(full_length, z)

        depth_dg["utmxs"] = utmxs.tolist()
        depth_dg["utmys"] = utmys.tolist()
        depth_dg["length"] = length.tolist()
        depth_dg["azimuth"] = azimuth.tolist()
        depth_dg["full_length"] = full_length.tolist()
        depth_dg["depression"] = depression.tolist()

        depth_dgs[n] = depth_dg

    return depth_dgs


def find_borders_db(depth):
    utmxs = []
    utmys = []

    for depth_dgs in depth:
        utmx = [
            [min(depth_dg["utmxs"]), max(depth_dg["utmxs"])] for depth_dg in depth_dgs
        ]
        utmy = [
            [min(depth_dg["utmys"]), max(depth_dg["utmys"])] for depth_dg in depth_dgs
        ]
        utmxs.extend(utmx)
        utmys.extend(utmy)

    min_x = np.amin(utmxs)
    max_x = np.amax(utmxs)
    dx = max_x - min_x
    min_y = np.amin(utmys)
    max_y = np.amax(utmys)
    dy = max_y - min_y

    info = {
        "min_x": min_x,
        "max_x": max_x,
        "dx": dx,
        "min_y": min_y,
        "max_y": max_y,
        "dy": dy,
    }

    return info


#################################################
# Criação das grades
#################################################


def create_grid_db(depth, pixel_size, no_val):
    info = find_borders_db(depth)

    # Largura e altura, com uma borda de 10 pixels para possivel interpolação
    W = int(info["dx"] / pixel_size) + 20
    H = int(info["dy"] / pixel_size) + 20

    # grid_bm = [[[] for y in range(W)] for x in range(H)]  # grid[y][x]
    # grid_bs = [[[] for y in range(W)] for x in range(H)]  # grid[y][x]

    min_val = np.min(
        [z for depth_dgs in depth for depth_dg in depth_dgs for z in depth_dg["z"]]
    )
    max_val = np.max(
        [z for depth_dgs in depth for depth_dg in depth_dgs for z in depth_dg["z"]]
    )
    n_bins = round((max_val - min_val) / 2)
    if n_bins < 100:
        n_bins = 100

    hist = DataBinningHist(
        rows=H,
        cols=W,
        min_val=min_val,
        max_val=max_val,
        n_bins=n_bins,
        min_smps=3,
        no_val=no_val,
    )

    for depth_dgs in depth:
        for depth_dg in depth_dgs:
            dist_x = np.subtract(depth_dg["utmxs"], info["min_x"])
            dist_y = np.subtract(depth_dg["utmys"], info["min_y"])

            cols = np.asarray((dist_x / info["dx"]) * (W - 20), dtype=np.uint32)
            lins = np.asarray((dist_y / info["dy"]) * (H - 20), dtype=np.uint32)
            cols += 10
            lins += 10
            cols[cols > (W - 10)] = W - 10
            lins[lins > (H - 10)] = H - 10

            for i, (col, lin) in enumerate(zip(cols, lins)):
                hist.add_value(lin, col, depth_dg["z"][i])
                # grid_bm[lin][col].append(depth_dg["z"][i])
                # grid_bs[lin][col].append(depth_dg["reflectivity"][i])

    # for lin in range(H):
    #     for col in range(W):
    #         if len(grid_bm[lin][col]) >= 3:
    #             grid_bm[lin][col] = np.median(grid_bm[lin][col])
    #         else:
    #             grid_bm[lin][col] = no_val
    #         if len(grid_bs[lin][col]) >= 3:
    #             grid_bs[lin][col] = np.median(grid_bs[lin][col])
    #         else:
    #             grid_bs[lin][col] = no_val

    grid = {
        # "bs": np.asarray(grid_bs),
        # "bm": np.asarray(grid_bm)
        "bm": hist.calc_median()
    }
    info["W"], info["H"] = W, H

    return grid, info


def create_bm_grid_db(depth_dgs, info, pixel_size, no_val):
    # Largura e altura, com uma borda de 10 pixels para possivel interpolação
    W = int(info["dx"] / pixel_size) + 20
    H = int(info["dy"] / pixel_size) + 20

    grid = [[[] for y in range(W)] for x in range(H)]  # grid[y][x]

    for depth_dg in depth_dgs:
        dist_x = np.subtract(depth_dg["utmxs"], info["min_x"])
        dist_y = np.subtract(depth_dg["utmys"], info["min_y"])

        cols = np.asarray((dist_x / info["dx"]) * (W - 20), dtype=np.uint32)
        lins = np.asarray((dist_y / info["dy"]) * (H - 20), dtype=np.uint32)
        cols += 10
        lins += 10
        cols[cols > (W - 10)] = W - 10
        lins[lins > (H - 10)] = H - 10

        for i, (col, lin) in enumerate(zip(cols, lins)):
            grid[lin][col].append(depth_dg["z"][i])

    for lin in range(H):
        for col in range(W):
            if len(grid[lin][col]) >= 3:
                grid[lin][col] = np.median(grid[lin][col])
            else:
                grid[lin][col] = no_val

    grid = np.asarray(grid)

    return grid


def create_waterfall_corr(sid_dgs, xyz_dgs, rra_dgs, pixel_size):
    no_val = -30000
    max_y = max(
        np.asarray(
            [np.ceil(max(np.absolute(xyz["y"])) / pixel_size) for xyz in xyz_dgs]
        )
    )
    W = max_y.astype(int) * 2 + 1

    wtfall_img = no_val * np.ones([len(sid_dgs), W])  # , np.int16)
    angles_img = no_val * np.ones([len(sid_dgs), W])  # , np.int16)
    for n, sid_dg in enumerate(sid_dgs):
        (
            smps,
            xs,
            ys,
            zs,
            travel_times,
            cnt_smps,
            reflectivity,
            Ns,
            smp_freq,
            sound_speed,
            ob_bs,
            ni_bs,
            Ro,
        ) = get_waterfall_info(sid_dg, xyz_dgs[n], rra_dgs[n])
        # Get pixel positions from Y values of central samples in beams
        waterfall_ys = np.round(ys / pixel_size + max_y).astype(int)
        # Get a range from min() to max() of pixel positions
        valid_interval = range(np.amin(waterfall_ys), np.amax(waterfall_ys) + 1)
        # Find travel times for this range by linear interpolation
        # of travel times as a function of pixel positions of central samples
        full_travel_times = np.interp(valid_interval, waterfall_ys, travel_times)
        # Find travel times' nearest neighbours 'cnt_travel_times'
        # to subtract from 'full_travel_times'
        f = interpolate.interp1d(waterfall_ys, travel_times, kind="nearest")
        cnt_travel_times = f(valid_interval)
        # Find nearest neighbours of 'cnt_smps'
        f = interpolate.interp1d(waterfall_ys, cnt_smps, kind="nearest")
        full_cnt_smps = f(valid_interval)
        # Go to sample numbers in beams
        sample_ns = np.subtract(full_travel_times, cnt_travel_times) * smp_freq
        sample_ns = np.add(sample_ns, full_cnt_smps)
        sample_ns = np.clip(sample_ns.astype(int), 0, len(smps) - 1)
        # Copy beams to waterfall
        wtfall_img[n, valid_interval] = smps[sample_ns]  # np.absolute(smps[sample_ns]

        # Calculate measured return angles for vectors xs, ys and zs
        cur_lengths = np.sqrt((xs**2) + (ys**2))
        cur_lengths[ys < 0] *= -1
        cur_measured_angles = np.arctan2(cur_lengths, zs) * 180 / math.pi
        angles_img[n, valid_interval] = np.round(
            np.interp(valid_interval, waterfall_ys, cur_measured_angles)
        )
    return waterfall_correction_by_angle2(wtfall_img, angles_img, no_val)


def waterfall_correction_by_angle2(wtfall_img, angles_img, no_val):
    window_size = 601
    angl_ref = 45
    H = wtfall_img.shape[0]
    W = wtfall_img.shape[1]
    un_angls = np.unique(angles_img)
    un_angls = un_angls[un_angls != no_val]
    angl_ref_i = np.where(un_angls == angl_ref)[0][0]
    avgs_by_angle = np.zeros((H, len(un_angls)))
    for i in range(H):
        # for k,an in enumerate(un_angls):
        avgs_by_angle[i, :] = [
            np.mean(wtfall_img[i, angles_img[i, :] == an]) for an in un_angls
        ]
        bad_idx = np.argwhere(np.isnan(avgs_by_angle[i, :]))
        good_idx = np.argwhere(~np.isnan(avgs_by_angle[i, :]))
        avgs_by_angle[i, bad_idx] = np.interp(
            bad_idx.flatten(), good_idx.flatten(), avgs_by_angle[i, good_idx].flatten()
        ).reshape(-1, 1)
    avgs_by_angle = signal.convolve2d(
        avgs_by_angle, np.ones((window_size, 1)) / window_size, "same"
    )
    # for i in range(len(un_angls)):
    #     avgs_by_angle[:, i] = signal.convolve(
    #     avgs_by_angle[:, i], np.ones(window_size) / window_size, "same"
    # )
    for i in range(H):
        avgs_by_angle[i, :] -= avgs_by_angle[i, angl_ref_i]
        # for k, an in enumerate(un_angls):
        #     cur_ok = np.where(angles_img[i, :] == an)
        #     out_wtfall_img[i, cur_ok] -= avgs_by_angle[i, k]
    return avgs_by_angle, un_angls


def create_sid_grids_from_wtfall(wtfall, xs, ys, info, pixel_size, no_val):
    # Largura e altura, com uma borda de 10 pixels para possivel interpolação
    W = int(info["dx"] / pixel_size) + 20
    H = int(info["dy"] / pixel_size) + 20

    min_val = wtfall[wtfall > no_val].min()
    max_val = wtfall.max()

    if min_val < -50:
        min_val = -50
    if max_val > 0:
        max_val = 0

    hist = DataBinningHist(
        rows=H,
        cols=W,
        min_val=db2amp(min_val),
        max_val=db2amp(max_val),
        n_bins=100,
        min_smps=3,
        no_val=no_val,
    )

    dist_x = xs.flatten() - info["min_x"]
    dist_y = ys.flatten() - info["min_y"]
    cols = np.asarray((dist_x / info["dx"]) * (W - 20), dtype=np.int32)
    lins = np.asarray((dist_y / info["dy"]) * (H - 20), dtype=np.int32)
    cols += 10
    lins += 10
    cols[cols < 0] = 0
    lins[lins < 0] = 0
    cols[cols > (W - 1)] = W - 1
    lins[lins > (H - 1)] = H - 1

    smps = wtfall.flatten()

    for smp, lin, col in zip(smps, lins, cols):
        hist.add_value(lin, col, db2amp(smp, no_val))
        # hist.add_value(lin, col, beam_smps[k])

    median = hist.calc_median()
    # median[median > no_val] += db2amp(-50)
    grid = amp2db(median, no_val)
    grid[np.isnan(grid)] = no_val

    return grid


def create_sid_grids_db(sid_dgs, depth_dgs, rra_dgs, info, pixel_size, no_val):
    # Largura e altura, com uma borda de 10 pixels para possivel interpolação
    W = int(info["dx"] / pixel_size) + 20
    H = int(info["dy"] / pixel_size) + 20

    avgs_by_angle, un_angls = create_waterfall_corr(sid_dgs, depth_dgs, rra_dgs, 1)

    smps = []
    cols = []
    lins = []

    for i, (sid_dg, depth_dg) in enumerate(zip(sid_dgs, depth_dgs)):
        smps_dg = []
        angls_dg = []
        samples = np.asarray(sid_dg["samples"], dtype=float) / 10
        Ns = np.asarray(sid_dg["Ns"])
        # Indices de onde começam as amostras (último indice é a quantidade total de amostras)
        i_smps = np.cumsum(Ns)
        i_smps = np.insert(i_smps, 0, 0)
        cnt_num = np.asarray(sid_dg["cnt_num"])  # Número da amostra central
        cnt_num -= 1  # Indice da amostra central

        utmx = depth_dg["utmx"]
        utmy = depth_dg["utmy"]
        z = np.asarray(depth_dg["z"])
        azimuth = np.asarray(depth_dg["azimuth"])
        full_length = np.asarray(depth_dg["full_length"])

        # Distância entre duas amostras
        smp_dist = depth_dg["sound_speed"] / depth_dg["smp_freq"]

        # ansls = []
        # ansls2 = []

        for j in range(len(i_smps) - 1):
            l, h = i_smps[j], i_smps[j + 1]
            c = cnt_num[j]

            beam_smps = samples[l:h]
            # Distâncias em relação à amostra central
            smp_dists = np.asarray(list(range(-c, h - l - c))) * smp_dist
            if sid_dg["sort_dir"][j] < 0:  # Sort direction = -1
                smp_dists *= -1

            full_lengths = full_length[j] + smp_dists
            lengths = np.sqrt((full_lengths**2) - (z[j] ** 2))
            dy = lengths * np.cos(azimuth[j])
            dx = lengths * np.sin(azimuth[j])
            utmxsmp = utmx + dx
            utmysmp = utmy + dy

            cur_lengths = lengths.copy()
            cur_lengths[dy < 0] *= -1
            cur_measured_angles = np.round(
                np.arctan2(cur_lengths, z[j]) * 180 / math.pi
            )

            dist_x = utmxsmp - info["min_x"]
            dist_y = utmysmp - info["min_y"]

            col = np.asarray((dist_x / info["dx"]) * (W - 20), dtype=np.int32)
            lin = np.asarray((dist_y / info["dy"]) * (H - 20), dtype=np.int32)
            col += 10
            lin += 10
            col[col < 0] = 0
            lin[lin < 0] = 0
            col[col > (W - 1)] = W - 1
            lin[lin > (H - 1)] = H - 1

            for k, (col, lin) in enumerate(zip(col, lin)):
                if np.isnan(beam_smps[k]):
                    continue
                smps_dg.append(beam_smps[k])
                angls_dg.append(cur_measured_angles[k])
                cols.append(col)
                lins.append(lin)
                # hist.add_value(lin, col, db2amp(beam_smps[k], no_val))
                # hist.add_value(lin, col, beam_smps[k])

        smps_dg = np.array(smps_dg)
        angls_dg = np.array(angls_dg)
        for k, an in enumerate(un_angls):
            cur_ok = np.where(angls_dg == an)
            smps_dg[cur_ok] -= avgs_by_angle[i, k]

        smps.extend(smps_dg)

    smps = np.array(smps)
    cols = np.array(cols)
    lins = np.array(lins)

    min_val = -55 if smps.min() < -55 else smps.min()
    max_val = -2 if smps.max() > -2 else smps.max()

    hist = DataBinningHist(
        rows=H,
        cols=W,
        min_val=db2amp(min_val),
        max_val=db2amp(max_val),
        n_bins=575,
        min_smps=3,
        no_val=no_val,
    )

    # valid_smps = (min_val < (smps-.1))& ((smps+.1)< max_val)

    # median = [[[] for _ in range(W)] for _ in range(H)]  # grid[y][x]

    # for s, c, l in zip(smps[valid_smps], cols[valid_smps], lins[valid_smps]):
    for s, c, l in zip(smps, cols, lins):
        if (min_val - 0.1) < s:  # and (max_val + 0.1) > s:
            hist.add_value(l, c, db2amp(s, no_val))
        # median[l][c].append(db2amp(s, no_val))

    # for lin in range(H):
    #     for col in range(W):
    #         if len(median[lin][col]) >= 3:
    #             median[lin][col] = np.median(median[lin][col])
    #         else:
    #             median[lin][col] = no_val

    median = hist.calc_median()
    # median[median > no_val] += db2amp(-50)
    grid = amp2db(median, no_val)
    grid[np.isnan(grid)] = no_val
    grid[grid == -55] = no_val
    # grid[grid < min_val + 0.1] = no_val

    return grid


def create_sid_grid_db(grid, no_val):
    grid = np.asarray(grid)
    H, W = grid.shape[1:3]
    grid = np.transpose(grid, (1, 2, 0)).tolist()

    for lin in range(H):
        for col in range(W):
            grid[lin][col] = [val for val in grid[lin][col] if val != no_val]
            if grid[lin][col]:
                grid[lin][col] = np.median(grid[lin][col])
            else:
                grid[lin][col] = no_val

    grid = np.asarray(grid)

    return grid


def dot_solar_vector_db(grid, solar_vector, bad_val):
    normal = np.ones(grid.shape) * bad_val

    for lin in range(1, grid.shape[0] - 1):
        for col in range(1, grid.shape[1] - 1):
            if grid[lin][col] != bad_val:
                a, b = grid[lin][col + 1], grid[lin - 1][col]
                c, d = grid[lin][col - 1], grid[lin + 1][col]
                p = np.asarray([col, lin, grid[lin][col]])
                pa = np.asarray([col + 1, lin, a])
                pb = np.asarray([col, lin - 1, b])
                pc = np.asarray([col - 1, lin, c])
                pd = np.asarray([col, lin + 1, d])
                if a != bad_val and b != bad_val:
                    cross_prod = np.cross(pa - p, pb - p)
                    normal[lin][col] = np.dot(cross_prod, solar_vector)
                elif b != bad_val and c != bad_val:
                    cross_prod = np.cross(pb - p, pc - p)
                    normal[lin][col] = np.dot(cross_prod, solar_vector)
                elif c != bad_val and d != bad_val:
                    cross_prod = np.cross(pc - p, pd - p)
                    normal[lin][col] = np.dot(cross_prod, solar_vector)
                elif d != bad_val and a != bad_val:
                    cross_prod = np.cross(pd - p, pa - p)
                    normal[lin][col] = np.dot(cross_prod, solar_vector)
            else:
                grid[lin][col] = bad_val

    return normal


def normalize_grid_db(grid, bad_val):
    min_val = grid[grid > bad_val].min()
    max_val = grid.max()

    grid[grid > bad_val] = (grid[grid > bad_val] - min_val) / (max_val - min_val)

    return grid


def generate_rgb_db(H, L, bad_val):
    import colorsys

    R = np.zeros(H.shape)
    G = np.zeros(H.shape)
    B = np.zeros(H.shape)

    for lin in range(H.shape[0]):
        for col in range(H.shape[1]):
            h = H[lin][col]
            l = L[lin][col]
            if h != bad_val and l != bad_val:
                r, g, b = colorsys.hls_to_rgb((2 / 3) * h, l, 0.8)
                R[lin][col] = r
                G[lin][col] = g
                B[lin][col] = b

    return np.asarray([R, G, B])


def interpolate_grid_db(grid, pixel_size, bad_val):
    # s_size = int(100 / pixel_size)
    # if s_size < 2:
    #     s_size = 2
    # elif s_size > 9:
    #     s_size = 9

    s_size = 7

    valid_pixels = grid != bad_val
    structure = np.ones((s_size, s_size)) == 1
    interp_pixels = ndimage.binary_closing(valid_pixels, structure=structure)

    points = []
    values = []
    interp_points = []

    for lin in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            if grid[lin][col] != bad_val:
                points.append((lin, col))
                values.append(grid[lin][col])
            if interp_pixels[lin][col]:
                interp_points.append((lin, col))

    interp_values = interpolate.griddata(
        points, values, interp_points, method="cubic", fill_value=bad_val
    )  # Interpola pontos

    interp_grid = np.ones(grid.shape) * bad_val
    for i, (lin, col) in enumerate(interp_points):
        interp_grid[lin][col] = interp_values[i]

    min_val = np.amin(grid[grid > bad_val])
    max_val = np.amax(grid)

    interp_grid[np.logical_and(interp_grid > bad_val, interp_grid < min_val)] = min_val
    interp_grid[interp_grid > max_val] = max_val

    return interp_grid


#################################################
# Arquivamento das imagens e carregamento no QGIS
#################################################


def get_convex_control_points_db(depth, img, bad_val, info):
    Ny = 40
    Nx = 40
    H = img.shape[0]
    W = img.shape[1]
    stepy = int(H / Ny) + 1
    stepx = int(W / Nx) + 1
    ys_img = list(range(0, H, stepy))
    ys_img[-1] = H - 1
    real_xy = []
    img_xy = []
    for y in ys_img:
        good_vals = np.nonzero(img[y, :] != bad_val)
        for x in good_vals[0][0:-1:stepx]:
            x_utm = (float(x - 10) / float(W - 20)) * info["dx"] + info["min_x"]
            y_utm = (float(y - 10) / float(H - 20)) * info["dy"] + info["min_y"]
            cur_real_xy = [x_utm, y_utm]
            cur_img_xy = [float(x), float(y)]
            real_xy.append(cur_real_xy)
            img_xy.append(cur_img_xy)
    gcpList = []
    for k in range(0, len(real_xy)):
        gcpList.append(
            gdal.GCP(real_xy[k][0], real_xy[k][1], 0, img_xy[k][0], img_xy[k][1])
        )
    return gcpList


def save_np_array_db(img_array, file_name):
    if img_array.dtype == "uint16":
        data_type = gdal.GDT_UInt16
    elif img_array.dtype == "int16":
        # print('OK')
        data_type = gdal.GDT_Int16
    elif img_array.dtype == "float32":
        data_type = gdal.GDT_Float32
    else:  # if img_array.dtype===='uint8':
        data_type = gdal.GDT_Byte

    # Create(file_name, width, height, num_bands, data_type)
    dst_ds = gdal.GetDriverByName("GTiff").Create(
        file_name, img_array.shape[1], img_array.shape[0], 1, data_type
    )
    dst_ds.GetRasterBand(1).WriteArray(img_array)


def save_np_array_rgb_db(img_array, file_name):
    if img_array.dtype == "uint16":
        data_type = gdal.GDT_UInt16
    elif img_array.dtype == "int16":
        # print('OK')
        data_type = gdal.GDT_Int16
    elif img_array.dtype == "float32":
        data_type = gdal.GDT_Float32
    else:  # if img_array.dtype===='uint8':
        data_type = gdal.GDT_Byte

    # Create(file_name, width, height, num_bands, data_type)
    dst_ds = gdal.GetDriverByName("GTiff").Create(
        file_name, img_array.shape[2], img_array.shape[1], 3, data_type
    )
    dst_ds.GetRasterBand(1).WriteArray(img_array[0])
    dst_ds.GetRasterBand(2).WriteArray(img_array[1])
    dst_ds.GetRasterBand(3).WriteArray(img_array[2])


def translate_warp_db(input_tif, input_points, epsg_code, output_tif):
    use_thinplate = True
    errorThreshold = 200

    in_tif = gdal.Open(input_tif)

    gtif = gdal.Translate(
        "",
        in_tif,
        GCPs=input_points,
        outputSRS=epsg_code,
        outputType=in_tif.GetRasterBand(1).DataType,
        noData=None,
        format="MEM",
    )
    if not output_tif:
        return gdal.Warp(
            "",
            gtif,
            format="MEM",
            srcSRS=epsg_code,
            dstSRS=epsg_code,
            tps=use_thinplate,
            errorThreshold=errorThreshold,
        )
    else:
        if os.path.isfile(output_tif):
            os.remove(output_tif)
        gdal.Warp(
            output_tif,
            gtif,
            format="GTiff",
            srcSRS=epsg_code,
            dstSRS=epsg_code,
            tps=use_thinplate,
            errorThreshold=errorThreshold,
        )


def remove_layers_db(layer_names):
    layers = [
        QgsProject.instance().mapLayersByName(layer_name) for layer_name in layer_names
    ]
    if layers:
        QgsProject.instance().removeMapLayers(
            [layer.id() for layer_list in layers for layer in layer_list]
        )


def load_image_QGIS_db(file_name, layer_name, layer_range, gray=True):
    rasterLyr = QgsRasterLayer(file_name, layer_name)
    if gray:
        myEnhancement = QgsContrastEnhancement(
            rasterLyr.renderer().dataType(rasterLyr.renderer().grayBand())
        )
        myEnhancement.setContrastEnhancementAlgorithm(
            QgsContrastEnhancement.StretchAndClipToMinimumMaximum, True
        )
        myEnhancement.setMinimumValue(float("{:.2f}".format(layer_range[0])))
        myEnhancement.setMaximumValue(float("{:.2f}".format(layer_range[1])))
        rasterLyr.renderer().setContrastEnhancement(myEnhancement)
    else:
        import colorsys

        n_colors = 10
        hs = np.linspace(0, 2 / 3, n_colors)
        colors = [
            "#"
            + hex(int(r * 255)).split("x")[-1]
            + hex(int(g * 255)).split("x")[-1]
            + hex(int(b * 255)).split("x")[-1]
            for (r, g, b) in (colorsys.hls_to_rgb(h, 0.5, 0.8) for h in hs)
        ]
        # colors = ['#e61919', '#e6a219', '#a2e619', '#19e619', '#19e6a1', '#19a1e6', '#1919e6']
        values = np.linspace(layer_range[0], layer_range[1], len(colors))
        lst = [
            QgsColorRampShader.ColorRampItem(
                value, QColor(color), "{:.2f}".format(value)
            )
            for (value, color) in zip(values, colors)
        ]

        myRasterShader = QgsRasterShader()
        myColorRamp = QgsColorRampShader()

        myColorRamp.setColorRampItemList(lst)
        myColorRamp.setColorRampType(QgsColorRampShader.Interpolated)
        myColorRamp.setClip(True)
        myRasterShader.setRasterShaderFunction(myColorRamp)

        myPseudoRenderer = QgsSingleBandPseudoColorRenderer(
            rasterLyr.dataProvider(), rasterLyr.type(), myRasterShader
        )
        myPseudoRenderer.setClassificationMin(layer_range[0])
        myPseudoRenderer.setClassificationMax(layer_range[1])
        rasterLyr.setRenderer(myPseudoRenderer)

    rasterLyr.isValid()
    QgsProject.instance().addMapLayers([rasterLyr])


def load_multiband_image_QGIS_db(file_name, layer_name):
    rasterLyr = QgsRasterLayer(file_name, layer_name)
    myEnhancement = QgsContrastEnhancement()
    myEnhancement.setContrastEnhancementAlgorithm(
        QgsContrastEnhancement.ClipToMinimumMaximum, True
    )
    myEnhancement.setMinimumValue(1)
    rasterLyr.renderer().setRedContrastEnhancement(myEnhancement)
    rasterLyr.renderer().setGreenContrastEnhancement(myEnhancement)
    rasterLyr.renderer().setBlueContrastEnhancement(myEnhancement)

    rasterLyr.isValid()
    QgsProject.instance().addMapLayers([rasterLyr])


# Definidas todas classes e funcoes, vamos rodar o plugin
class ExAlgo(QgsProcessingAlgorithm):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

    def __init__(self):
        super().__init__()

    def name(self):
        return "EchoGIS"

    def tr(self, text):
        return QCoreApplication.translate("EchoGIS_stat", text)

    def displayName(self):
        return self.tr("EchoGIS_stat")

    def group(self):
        return self.tr("EchoGIS_Plugin_stat")

    def groupId(self):
        return "echogis_plugin_stat"

    def shortHelpString(self):
        return self.tr(
            "QGIS plugin for the inversion of sea-bottom attributes, developed by"
            " profs. Luciano Emídio Neves da Fonseca and Diogo Caetano Garcia, from the"
            " University of Brasília, Brazil."
        )

    def helpUrl(self):
        return ""

    def createInstance(self):
        return type(self)()

    def initAlgorithm(self, config=None):
        # print('Load plugin')
        self.input = "INPUT"

    def processAlgorithm(self, parameters, context, feedback):
        # print('Start plugin')
        buttons = Buttons()
        buttons.show()
        return {self.OUTPUT: "ok"}
