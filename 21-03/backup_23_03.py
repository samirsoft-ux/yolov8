    #CALCULAR EL CHOQUE DEL TACO CON UNA BOLA
    def colision_trayectoria(self, frame, centro_suavizado, direccion_suavizada, centros_bolas, radio_bolas, grosor_taco):
        punto_colision_cercano = None
        distancia_minima = float('inf')
        # Ajusta el radio de las bolas para considerar el grosor del taco
        radio_efectivo = radio_bolas + grosor_taco / 2
        
        for centro_bola in centros_bolas:
            # Calcula el punto en la trayectoria del taco más cercano al centro de la bola
            punto_mas_cercano, distancia_al_centro = self.calcular_punto_mas_cercano_y_distancia(centro_suavizado, direccion_suavizada, centro_bola)
            
            if distancia_al_centro <= radio_efectivo:
                # Verifica si este es el punto de colisión más cercano encontrado hasta ahora
                if distancia_al_centro < distancia_minima:
                    distancia_minima = distancia_al_centro
                    # Aquí, en lugar de simplemente seleccionar el punto_mas_cercano,
                    punto_interseccion_int = tuple(np.int32(punto_mas_cercano))
                    
                    ##PRUEBA PARA VER EL PUNTO DE CHOQUE PERO DE LA LINEA PERPENDICULAR
                    #cv2.circle(frame, punto_interseccion_int, 20, (0, 255, 255), -1)  # Dibuja un círculo amarillo
        
                    # utilizamos la función encontrar_punto_interseccion para obtener una posición de colisión más precisa.
                    punto_interseccion_exacto = self.encontrar_punto_interseccion(frame, centro_suavizado, direccion_suavizada, centro_bola, radio_bolas)
                    
                    # Es posible que punto_interseccion_exacto sea None si no hay intersección real debido al discriminante negativo.
                    if punto_interseccion_exacto is not None:
                        punto_colision_cercano = punto_interseccion_exacto
                        # Actualiza la distancia mínima con la distancia al punto de intersección exacto
                        distancia_minima = np.linalg.norm(centro_suavizado - punto_interseccion_exacto)

        if punto_colision_cercano is not None:
            # Verifica si el punto de colisión está suficientemente cerca del centro de la bola
            umbral_proximidad = 5  # Ajusta este valor según sea necesario
            distancia_al_centro_bola = np.linalg.norm(np.array(centro_bola) - np.array(punto_interseccion_int))
            
            if distancia_al_centro_bola <= umbral_proximidad:
                # Calcular la dirección y velocidad de la bola post-impacto
                vector_velocidad_bola = self.calcular_vector_velocidad_bola_post_impacto(direccion_suavizada, punto_colision_cercano, centros_bolas[0])
                
                # Calcular el punto opuesto al punto de colisión, a través del centro de la bola
                # Esto crea un vector desde el punto de colisión hacia el centro de la bola
                vector_al_centro = np.array(centro_bola) - np.array(punto_colision_cercano)
                # Normalizar este vector
                vector_al_centro_normalizado = vector_al_centro / np.linalg.norm(vector_al_centro)
                # Calcular el punto inicial y final de la trayectoria proyectada
                punto_inicio_trayectoria = np.array(centro_bola) + vector_al_centro_normalizado * radio_bolas
                
                # Configuraciones iniciales para la simulación de rebotes
                velocidad_fija_bola = 1500  # Velocidad inicial arbitraria
                coeficiente_friccion = 0.9  # Factor de reducción de velocidad por rebote
                rebotes_contador = 0  # Contador de rebotes
                
                # Inicia la simulación de la trayectoria de la bola desde el punto de impacto
                punto_actual = punto_inicio_trayectoria
                direccion_actual = vector_velocidad_bola / np.linalg.norm(vector_velocidad_bola)

                # Mientras la "velocidad" de la bola no sea insignificante
                while velocidad_fija_bola > 1.0 and rebotes_contador < 2:
                    # Calcula el punto final proyectado
                    punto_final_proyectado = punto_actual + direccion_actual * velocidad_fija_bola

                    # Intenta detectar una colisión con las bandas de la mesa
                    punto_colision_banda, segmento_colision = self.detectar_colision_trayectoria_con_banda(punto_actual, punto_final_proyectado, self.mesa_corners)

                    if punto_colision_banda:
                        # Si detecta una colisión, calcula la normal de la banda
                        normal_banda = self.calcular_normal_banda(segmento_colision[0], segmento_colision[1])
                                                
                        # Refleja la dirección de la trayectoria basándose en la normal de la banda
                        direccion_actual = self.calcular_vector_reflexion(direccion_actual, normal_banda)
                        
                        # Reduce la velocidad debido al rebote
                        velocidad_fija_bola *= coeficiente_friccion
                        
                        # Dibuja la trayectoria desde el punto actual hasta el punto de colisión
                        cv2.line(frame, tuple(np.int32(punto_actual)), tuple(np.int32(punto_colision_banda)), (0, 255, 0), 2)

                        # Actualiza el punto actual para el siguiente cálculo
                        punto_actual = punto_colision_banda
                        
                        rebotes_contador += 1  # Incrementa el contador de rebotes
                    else:
                        break  # Termina el bucle
            
            return True, punto_colision_cercano
        else:
            return False, None