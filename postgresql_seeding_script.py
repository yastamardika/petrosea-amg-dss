import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import random
from faker import Faker
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    'host': '10.0.6.3',
    'database': 'petro',
    'user': 'user-name',
    'password': 'strong-password',
    'port': '5432'
}

class DatabaseSeeder:
    def __init__(self, db_config):
        self.db_config = db_config
        self.fake = Faker()
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from database"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Disconnected from database")
    
    def execute_sql_file(self, sql_file_path):
        """Execute SQL file to create tables"""
        try:
            with open(sql_file_path, 'r') as file:
                sql_content = file.read()
                self.cursor.execute(sql_content)
                self.conn.commit()
                logger.info(f"Successfully executed SQL file: {sql_file_path}")
        except Exception as e:
            logger.error(f"Error executing SQL file: {e}")
            self.conn.rollback()
            raise
    
    def seed_time_dimension(self, start_date='2023-01-01', end_date='2025-12-31'):
        """Seed time dimension table"""
        logger.info("Seeding time dimension...")
        
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        # Holiday dates (example)
        holidays = {
            '2024-01-01': 'New Year Day',
            '2024-12-25': 'Christmas Day',
            '2024-07-04': 'Independence Day',
            '2024-11-28': 'Thanksgiving'
        }
        
        current_date = start
        batch_data = []
        
        while current_date <= end:
            is_weekday = current_date.weekday() < 5
            holiday_name = holidays.get(current_date.strftime('%Y-%m-%d'))
            is_holiday = holiday_name is not None
            
            batch_data.append((
                current_date,
                current_date.strftime('%A'),
                current_date.day,
                current_date.timetuple().tm_yday,
                current_date.month,
                current_date.strftime('%B'),
                (current_date.month - 1) // 3 + 1,
                current_date.year,
                is_weekday,
                is_holiday,
                holiday_name
            ))
            
            current_date += timedelta(days=1)
            
            # Insert in batches of 1000
            if len(batch_data) >= 1000:
                self._insert_time_batch(batch_data)
                batch_data = []
        
        # Insert remaining data
        if batch_data:
            self._insert_time_batch(batch_data)
        
        logger.info(f"Time dimension seeded with {(end - start).days + 1} records")
    
    def _insert_time_batch(self, batch_data):
        """Insert batch of time dimension data"""
        insert_sql = """
        INSERT INTO dim_time (date_actual, day_of_week, day_number_in_month, 
                             day_number_in_year, month_actual, month_name, 
                             quarter_actual, year_actual, is_weekday, is_holiday, holiday_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (date_actual) DO NOTHING
        """
        self.cursor.executemany(insert_sql, batch_data)
        self.conn.commit()
    
    def seed_departments(self):
        """Seed department dimension"""
        logger.info("Seeding departments...")
        
        departments = [
            ('Heavy Equipment Operations', 'HEO', None),
            ('Maintenance & Repair', 'MNT', None),
            ('Safety & Environment', 'SHE', None),
            ('Engineering', 'ENG', None),
            ('Operations Control', 'OPS', None),
            ('Quality Assurance', 'QAS', None),
            ('Human Resources', 'HRS', None),
            ('Administration', 'ADM', None),
            ('IT Support', 'ITS', None),
            ('Logistics', 'LOG', None)
        ]
        
        for dept_name, dept_code, parent_id in departments:
            self.cursor.execute("""
                INSERT INTO dim_department (department_name, department_code, parent_department_id, valid_from)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (department_name, department_code, valid_from) DO NOTHING
            """, (dept_name, dept_code, parent_id, date.today()))
        
        self.conn.commit()
        logger.info(f"Departments seeded with {len(departments)} records")
    
    def seed_positions(self):
        """Seed position dimension"""
        logger.info("Seeding positions...")
        
        positions = [
            ('Heavy Equipment Operator', 'Operations', 'Senior', False),
            ('Equipment Technician', 'Technical', 'Mid', False),
            ('Safety Officer', 'Safety', 'Senior', False),
            ('Maintenance Supervisor', 'Technical', 'Senior', True),
            ('Operations Manager', 'Management', 'Executive', True),
            ('Quality Inspector', 'Quality', 'Mid', False),
            ('Site Engineer', 'Engineering', 'Senior', False),
            ('HR Specialist', 'Administrative', 'Mid', False),
            ('IT Technician', 'Technical', 'Junior', False),
            ('Logistics Coordinator', 'Operations', 'Mid', False),
            ('Training Coordinator', 'Administrative', 'Mid', False),
            ('Environmental Specialist', 'Safety', 'Senior', False)
        ]
        
        for pos_name, pos_category, pos_level, is_supervisory in positions:
            self.cursor.execute("""
                INSERT INTO dim_position (position_name, position_category, position_level, 
                                        is_supervisory, valid_from)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (position_name, valid_from) DO NOTHING
            """, (pos_name, pos_category, pos_level, is_supervisory, date.today()))
        
        self.conn.commit()
        logger.info(f"Positions seeded with {len(positions)} records")
    
    def seed_locations(self):
        """Seed location dimension"""
        logger.info("Seeding locations...")
        
        locations = [
            ('Main Site', 'MAIN', 'Primary', 'Jakarta', 'Jakarta', 'Indonesia'),
            ('North Pit', 'NPIT', 'Mining Area', 'Balikpapan', 'East Kalimantan', 'Indonesia'),
            ('South Pit', 'SPIT', 'Mining Area', 'Balikpapan', 'East Kalimantan', 'Indonesia'),
            ('Workshop Area', 'WSHP', 'Maintenance', 'Balikpapan', 'East Kalimantan', 'Indonesia'),
            ('Office Complex', 'OFFC', 'Administrative', 'Jakarta', 'Jakarta', 'Indonesia'),
            ('Storage Facility', 'STOR', 'Warehouse', 'Surabaya', 'East Java', 'Indonesia')
        ]
        
        for loc_name, loc_code, loc_type, city, region, country in locations:
            self.cursor.execute("""
                INSERT INTO dim_location (location_name, location_code, location_type, 
                                        city, region, country, valid_from)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (location_name, location_code, valid_from) DO NOTHING
            """, (loc_name, loc_code, loc_type, city, region, country, date.today()))
        
        self.conn.commit()
        logger.info(f"Locations seeded with {len(locations)} records")
    
    def seed_shift_types(self):
        """Seed shift type dimension"""
        logger.info("Seeding shift types...")
        
        shifts = [
            ('D', 'Day Shift', time(7, 0), time(19, 0), 12.0, False),
            ('N', 'Night Shift', time(19, 0), time(7, 0), 12.0, True),
            ('M', 'Morning Shift', time(6, 0), time(14, 0), 8.0, False),
            ('A', 'Afternoon Shift', time(14, 0), time(22, 0), 8.0, False),
            ('E', 'Evening Shift', time(22, 0), time(6, 0), 8.0, True)
        ]
        
        for shift_code, shift_name, start_time, end_time, duration, is_night in shifts:
            self.cursor.execute("""
                INSERT INTO dim_shift_type (shift_code, shift_name, shift_start_time, 
                                          shift_end_time, shift_duration, is_night_shift)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (shift_code) DO NOTHING
            """, (shift_code, shift_name, start_time, end_time, duration, is_night))
        
        self.conn.commit()
        logger.info(f"Shift types seeded with {len(shifts)} records")
    
    def seed_employees(self, num_employees=50):
        """Seed employee dimension"""
        logger.info(f"Seeding {num_employees} employees...")
        
        # Get existing dimension data
        self.cursor.execute("SELECT department_id FROM dim_department WHERE is_active = TRUE")
        dept_ids = [row['department_id'] for row in self.cursor.fetchall()]
        
        self.cursor.execute("SELECT position_id FROM dim_position WHERE is_active = TRUE")
        pos_ids = [row['position_id'] for row in self.cursor.fetchall()]
        
        self.cursor.execute("SELECT location_id FROM dim_location WHERE is_active = TRUE")
        loc_ids = [row['location_id'] for row in self.cursor.fetchall()]
        
        employees = []
        for i in range(num_employees):
            emp_id = f"EMP{i+1:04d}"
            name = self.fake.name()
            gender = random.choice(['Male', 'Female'])
            dob = self.fake.date_of_birth(minimum_age=22, maximum_age=60)
            hire_date = self.fake.date_between(start_date='-5y', end_date='today')
        
            name_parts = name.split()
            first_name = name_parts[0].lower() if len(name_parts) > 0 else "user"
            last_name = name_parts[-1].lower() if len(name_parts) > 1 else "employee"
        
            email_base = f"{first_name}.{last_name}"[:10]  # Limit base to 15 chars
            email = f"{email_base}@company.com"
            phone = self.fake.phone_number()[:12]  # Limit phone number to 12 chars
            dept_id = random.choice(dept_ids)
            pos_id = random.choice(pos_ids)
            loc_id = random.choice(loc_ids)
            supervisor_id = f"EMP{random.randint(1, max(1, i)):04d}" if i > 0 else None
            status = random.choice(['Active', 'Active', 'Active', 'On-Leave'])  # More active employees
            
            employees.append((
                emp_id, name, gender, dob, hire_date, email, phone,
                dept_id, pos_id, loc_id, supervisor_id, status, date.today()
            ))
        
        insert_sql = """
        INSERT INTO dim_employee (employee_id, employee_name, gender, date_of_birth, 
                                date_of_hire, email, phone, department_id, position_id, 
                                location_id, supervisor_id, employee_status, valid_from)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (employee_id, valid_from) DO NOTHING
        """
        
        self.cursor.executemany(insert_sql, employees)
        self.conn.commit()
        logger.info(f"Employees seeded with {num_employees} records")
    
    def seed_performance_data(self, days_back=90):
        """Seed performance fact data"""
        logger.info(f"Seeding performance data for last {days_back} days...")
        
        # Get active employees
        self.cursor.execute("""
            SELECT employee_key, employee_id FROM dim_employee 
            WHERE is_current = TRUE AND employee_status = 'Active'
        """)
        employees = self.cursor.fetchall()
        
        # Get time dimension data
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        self.cursor.execute("""
            SELECT time_id, date_actual FROM dim_time 
            WHERE date_actual BETWEEN %s AND %s
        """, (start_date, end_date))
        time_data = self.cursor.fetchall()
        
        performance_records = []
        
        for emp in employees[:20]:  # Limit to first 20 employees for performance
            for time_record in random.sample(time_data, min(30, len(time_data))):  # Random 30 days
                
                # Generate performance metrics based on employee consistency
                base_performance = random.uniform(70, 95)
                variation = random.uniform(-10, 10)
                
                performance_records.append((
                    emp['employee_key'],
                    time_record['time_id'],
                    time_record['date_actual'] - timedelta(days=30),  # eval period start
                    time_record['date_actual'],  # eval period end
                    max(60, min(100, base_performance + variation)),  # bipo_compliance
                    max(60, min(100, base_performance + random.uniform(-5, 5))),  # performance_rating
                    max(60, min(100, base_performance + random.uniform(-8, 8))),  # quality_score
                    max(60, min(100, base_performance + random.uniform(-6, 6))),  # productivity_score
                    max(60, min(100, base_performance + random.uniform(-4, 4))),  # safety_score
                    max(60, min(100, base_performance + random.uniform(-7, 7))),  # attendance_score
                    random.choice(['EMP0001', 'EMP0002', 'EMP0003']),  # evaluator_id
                    True,  # is_approved
                    self.fake.text(max_nb_chars=200),  # comments
                    'HRIS_SYSTEM'  # source_system
                ))
        
        insert_sql = """
        INSERT INTO fact_performance (employee_key, date_key, evaluation_period_start, 
                                    evaluation_period_end, bipo_compliance_percentage, 
                                    performance_rating, quality_score, productivity_score, 
                                    safety_score, attendance_score, evaluator_id, 
                                    is_approved, comments, source_system)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        self.cursor.executemany(insert_sql, performance_records)
        self.conn.commit()
        logger.info(f"Performance data seeded with {len(performance_records)} records")
    
    def seed_attendance_data(self, days_back=90):
        """Seed attendance fact data"""
        logger.info(f"Seeding attendance data for last {days_back} days...")
        
        # Get active employees
        self.cursor.execute("""
            SELECT employee_key FROM dim_employee 
            WHERE is_current = TRUE AND employee_status = 'Active'
        """)
        employees = [row['employee_key'] for row in self.cursor.fetchall()]
        
        # Get shift types
        self.cursor.execute("SELECT shift_type_id FROM dim_shift_type WHERE is_active = TRUE")
        shift_types = [row['shift_type_id'] for row in self.cursor.fetchall()]
        
        # Get time dimension data
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        self.cursor.execute("""
            SELECT time_id, date_actual, is_weekday FROM dim_time 
            WHERE date_actual BETWEEN %s AND %s
        """, (start_date, end_date))
        time_data = self.cursor.fetchall()
        
        attendance_records = []
        
        for emp_key in employees:
            for time_record in time_data:
                # Skip some weekend days
                if not time_record['is_weekday'] and random.random() < 0.7:
                    continue
                
                # Random absence (5% chance)
                is_absent = random.random() < 0.05
                
                if not is_absent:
                    shift_id = random.choice(shift_types)
                    base_time = datetime.combine(time_record['date_actual'], time(7, 0))
                    
                    # Add some variation to arrival time
                    time_in = base_time + timedelta(minutes=random.randint(-10, 30))
                    time_out = time_in + timedelta(hours=random.uniform(8, 12))
                    
                    hours_worked = (time_out - time_in).total_seconds() / 3600
                    is_late = (time_in.time() > time(7, 15))
                    minutes_late = max(0, int((time_in.time().hour * 60 + time_in.time().minute) - (7 * 60 + 15))) if is_late else 0
                    is_overtime = hours_worked > 8
                    overtime_hours = max(0, hours_worked - 8) if is_overtime else 0
                    
                    attendance_records.append((
                        emp_key, time_record['time_id'], shift_id, time_in, time_out,
                        hours_worked, False, is_late, minutes_late, is_overtime,
                        overtime_hours, True, None, 'ATTENDANCE_SYSTEM'
                    ))
                else:
                    # Absent record
                    attendance_records.append((
                        emp_key, time_record['time_id'], None, None, None,
                        0, True, False, 0, False, 0, True,
                        random.choice(['Sick Leave', 'Personal Leave', 'Emergency']),
                        'ATTENDANCE_SYSTEM'
                    ))
        
        insert_sql = """
        INSERT INTO fact_attendance (employee_key, date_key, shift_type_id, time_in, time_out,
                                   hours_worked, is_absent, is_late, minutes_late, is_overtime,
                                   overtime_hours, is_approved, absence_reason, source_system)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        self.cursor.executemany(insert_sql, attendance_records)
        self.conn.commit()
        logger.info(f"Attendance data seeded with {len(attendance_records)} records")
    
    def seed_sleep_metrics(self, days_back=90):
        """Seed sleep metrics fact data"""
        logger.info(f"Seeding sleep metrics for last {days_back} days...")
        
        # Get active employees (subset for sleep tracking)
        self.cursor.execute("""
            SELECT employee_key FROM dim_employee 
            WHERE is_current = TRUE AND employee_status = 'Active'
            LIMIT 30
        """)
        employees = [row['employee_key'] for row in self.cursor.fetchall()]
        
        # Get time dimension data
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        self.cursor.execute("""
            SELECT time_id FROM dim_time 
            WHERE date_actual BETWEEN %s AND %s
        """, (start_date, end_date))
        time_ids = [row['time_id'] for row in self.cursor.fetchall()]
        
        sleep_records = []
        
        for emp_key in employees:
            for time_id in random.sample(time_ids, min(60, len(time_ids))):  # Random 60 days
                sleep_time = time(random.randint(21, 23), random.randint(0, 59))
                wake_time = time(random.randint(5, 7), random.randint(0, 59))
                
                # Calculate sleep duration
                sleep_duration = random.uniform(6, 9)
                fixed_sleep_duration = max(6, sleep_duration - 0.5)  # Adjusted sleep duration
                sleep_quality = random.randint(6, 10)
                
                sleep_records.append((
                    emp_key, time_id, sleep_time, wake_time, sleep_duration,
                    fixed_sleep_duration, sleep_quality, 'SLEEP_TRACKER'
                ))
        
        insert_sql = """
        INSERT INTO fact_sleep_metrics (employee_key, date_key, sleep_time, wake_time,
                                      sleep_duration_hours, fixed_sleep_duration_hours,
                                      sleep_quality_rating, source_system)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        self.cursor.executemany(insert_sql, sleep_records)
        self.conn.commit()
        logger.info(f"Sleep metrics seeded with {len(sleep_records)} records")
    
    def seed_all(self):
        """Seed all dimension and fact tables"""
        logger.info("Starting full database seeding...")
        
        try:
            self.connect()
            
            # Seed dimensions first
            self.seed_time_dimension()
            self.seed_departments()
            self.seed_positions()
            self.seed_locations()
            self.seed_shift_types()
            self.seed_employees()
            
            # Seed fact tables
            self.seed_performance_data()
            self.seed_attendance_data()
            self.seed_sleep_metrics()
            
            logger.info("Database seeding completed successfully!")
            
        except Exception as e:
            logger.error(f"Error during seeding: {e}")
            if self.conn:
                self.conn.rollback()
            raise
        finally:
            self.disconnect()

def main():
    """Main function to run the seeding process"""
    print("Employee Data Mart - Database Seeding")
    print("="*50)
    
    # Update DB_CONFIG with your actual database credentials
    print("Please update DB_CONFIG with your database credentials before running!")
    print(f"Current config: {DB_CONFIG}")
    
    # confirm = input("\nDo you want to proceed with seeding? (yes/no): ")
    confirm = "yes"
    if confirm.lower() not in ['yes', 'y']:
        print("Seeding cancelled.")
        return
    
    seeder = DatabaseSeeder(DB_CONFIG)
    
    try:
        seeder.seed_all()
        print("\n✅ Database seeding completed successfully!")
        print("\nNext steps:")
        print("1. Update the Streamlit app with your database credentials")
        print("2. Run the Streamlit app: streamlit run employee_dss_app.py")
        
    except Exception as e:
        print(f"\n❌ Seeding failed: {e}")
        print("Please check your database connection and try again.")

if __name__ == "__main__":
    main()