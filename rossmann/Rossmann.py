import pickle
import inflection
import pandas as pd
import numpy as np
import datetime

class Rossmann(object):
    def __init__(self):
        self.home_path = ''
        self.competition_distance_scaler   = pickle.load(open(self.home_path +'parameter/competition_distance_scaler.pkl', 'rb'))
        self.competition_time_month_scaler = pickle.load(open(self.home_path +'parameter/competition_time_month_scaler.pkl', 'rb'))
        self.promo_time_week_scaler        = pickle.load(open(self.home_path +'parameter/promo_time_week_scaler.pkl', 'rb'))
        self.year_scaler                   = pickle.load(open(self.home_path +'parameter/year_scaler.pkl', 'rb'))
        self.store_type_scaler             = pickle.load(open(self.home_path +'parameter/store_type_scaler.pkl', 'rb'))
        
    def data_cleaning(self, df1):    
            
        ## 1.1. Rename Columns

        snakecase = lambda x: inflection.underscore(x)
        cols_new = list(map(snakecase,df1.columns))
        df1.columns = cols_new

        ## 1.3. Data Types
        df1['date'] = pd.to_datetime(df1['date'])
        
        ### 1.4.2 Fillout NA

        # competition_distance 
        ##coloca um valor extremamente alta se o competition distance for NaN
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 200000.0 if np.isnan(x) else x)

        # competition_open_since_month
        ##Se o valor for NaN substitui o mês localizado na coluna date
        filtro_mes = lambda x: x['date'].month if np.isnan(x['competition_open_since_month']) else x['competition_open_since_month']
        df1['competition_open_since_month'] = df1.apply(filtro_mes, axis=1) 

         # competition_open_since_year 
        filtro_ano = lambda x: x['date'].year if np.isnan(x['competition_open_since_year']) else x['competition_open_since_year']
        df1['competition_open_since_year'] = df1.apply(filtro_ano, axis=1)              

        # promo2_since_week
        filtro_promo_week = lambda x: x['date'].week if np.isnan(x['promo2_since_week']) else x['promo2_since_week']
        df1['promo2_since_week'] = df1.apply(filtro_promo_week, axis=1)    

        # promo2_since_year
        filtro_promo_year = lambda x: x['date'].year if np.isnan(x['promo2_since_year']) else x['promo2_since_year']
        df1['promo2_since_year'] = df1.apply(filtro_promo_year, axis=1)

        # promo_interval 
        df1['promo_interval'].fillna(0,inplace=True)

        months = { 1: 'Jan',
                   2: 'Feb',
                   3: 'Mar',
                   4: 'Apr',
                   5: 'May',
                   6: 'Jun',
                   7: 'Jul',
                   8: 'Aug',
                   9: 'Sept',
                   10: 'Oct',
                   11: 'Nov',
                   12: 'Dec',
                 }
        ##cria um coluna month_map com o nome do mês baseado na coluna date
        df1['month_map'] = df1['date'].apply(lambda x: x.month)
        def numero_em_mes(mes):
          return months[mes]
        df1['month_map'] = df1['month_map'].apply(numero_em_mes) 
        ##cria uma coluna is_promo: se o mês do month_map estiver dentro da coluna promo_interval, retorna 1, senão 0
        filtro_promo_map = lambda x: 1 if x['month_map'] in x['promo_interval'].split(',') else 0
        df1['is_promo'] = df1.apply(lambda x: (1 if x['month_map'] in x['promo_interval'].split(',') else 0) if x['promo_interval'] != 0 else 0, axis=1)


        ## 1.5. Change Types

        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)

        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)
        
        return df1
    
    def feature_engineering(self, df2):
    
        #year
        df2['year'] = df2['date'].dt.year
        #month
        df2['month'] = df2['date'].dt.month
        #day
        df2['day'] = df2['date'].dt.day
        #week of year
        df2['week_of_year'] = df2['date'].dt.weekofyear
        #year week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        #competition since
        ##Junta as duas colunas em uma só
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'],month=x['competition_open_since_month'],day=1), axis=1)

        #promo since
        #Junta as duas colunas em uma só transformando a semana em mês
        df2['promo_since'] = df2.apply(lambda x: datetime.datetime.strptime(f"{x['promo2_since_year']}-{x['promo2_since_week']-1}-1", '%Y-%W-%w'),axis=1)

        #assortment
        assortments = {'a' : 'basic', 'b' : 'extra', 'c' : 'extended'}
        def name_assortsments(valor):
          return assortments[valor]

        df2['assortment'] = df2['assortment'].apply(name_assortsments)

        #state holiday 
        holidays = {'a' : 'public_holiday', 'b' : 'easter_holiday', 'c' : 'christmas', '0' : 'regular_day'}
        def name_holidays(valor):
          return holidays[valor]

        df2['state_holiday'] = df2['state_holiday'].apply(name_holidays)

        #promo_time_week signifca quantos dias apos o pedido a promo2 está ativa(estendida)
        df2['promo_time_week'] = ( ( df2['date'] - df2['promo_since'] )/7 ).apply( lambda x: x.days ).astype( int )
        df2['competition_time_month'] = ( ( df2['date'] - df2['competition_since'] )/30 ).apply( lambda x: x.days ).astype( int )

        df2 = df2.loc[df2['open'] != 0]
        cols_drop = ['open','promo_interval','month_map']
        df2 = df2.drop(cols_drop, axis=1)
        
        return df2
    
    def data_preparation(self,df5):

        ## 5.2 Rescalling

        #competition_distance
        #tem que usar o reshape porque o robust scaler pede
        df5['competition_distance'] = self.competition_distance_scaler.transform(df5['competition_distance'].values.reshape(-1,1))
        

        #competition_time_month
        df5['competition_time_month'] = self.competition_time_month_scaler.transform(df5['competition_time_month'].values.reshape(-1,1))
        

        #promo_time_week
        df5['promo_time_week'] = self.promo_time_week_scaler.transform(df5['promo_time_week'].values.reshape(-1,1))
        

        #year
        df5['year'] = self.year_scaler.transform(df5['year'].values.reshape(-1,1))
        

        ## 5.3.1 Encoding

        #state_holiday - One Hot Encoding
        df5 = pd.get_dummies(df5, prefix=['state_holiday'], columns=['state_holiday'])

        df5['store_type'] = self.store_type_scaler.transform(df5['store_type'])
        


        #assortment
        types_assortment = {'basic': 1,
                            'extra': 2,
                            'extended': 3}
        df5['assortment'] = df5['assortment'].map(types_assortment)

        ## 5.3.3 Nature Transformation

        #day_of_week
        df5['day_of_week_cos'] = df5['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi/7)))
        df5['day_of_week_sen'] = df5['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi/7)))
        #day
        df5['day_sen'] = df5['day'].apply(lambda x: np.sin(x * (2. * np.pi/30)))
        df5['day_cos'] = df5['day'].apply(lambda x: np.cos(x * (2. * np.pi/30)))
        #week_of_year
        df5['week_of_year_sen'] = df5['week_of_year'].apply(lambda x: np.sin(x * (2. * np.pi/52)))
        df5['week_of_year_cos'] = df5['week_of_year'].apply(lambda x: np.cos(x * (2. * np.pi/52)))
        #month
        df5['month_sen'] = df5['month'].apply(lambda x: np.sin(x * (2. * np.pi/12)))
        df5['month_cos'] = df5['month'].apply(lambda x: np.cos(x * (2. * np.pi/12))) 

        cols_selected = [
         'store',
         'promo',
         'store_type',
         'assortment',
         'competition_distance',
         'competition_open_since_month',
         'competition_open_since_year',
         'promo2',
         'promo2_since_week',
         'promo2_since_year',
         'promo_time_week',
         'competition_time_month',
         'day_of_week_cos',
         'day_of_week_sen',
         'day_sen',
         'day_cos',
         'week_of_year_cos',
         'week_of_year_sen',
         'month_sen',
         'month_cos']
        
        return df5[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        #prediction
        pred = model.predict(test_data)
        
        #join pred into the original data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient='records', date_format='iso')
        