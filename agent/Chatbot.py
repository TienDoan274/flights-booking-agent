from dotenv import load_dotenv
import os
import sys

# set your OPENAI_API_KEY in .env file
parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)
from constant import *

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(dotenv_path)

import prompts
from llama_index.llms.openai import OpenAI
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    step, 
    Event,
    Context,
    Workflow,
    draw_all_possible_flows,
    WorkflowCheckpointer
)
import dateparser
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.prompts import PromptTemplate
from pymongo import MongoClient
from pydantic import BaseModel
from enum import Enum
import json
from bson import ObjectId

class FlightSchema(BaseModel):
    date:Optional[str] = None
    scheduled_time:Optional[str] = None
    flight_id:Optional[str] = None
    counter:Optional[str] = None
    gate:Optional[str] = None
    departure_region_name:Optional[str] = None
    arrival_region_name:Optional[str] = None
    departure_airport:Optional[str] = None
    arrival_airport:Optional[str] = None
    airline:Optional[str] = None

'''
schema = StructType([
        StructField("departure_time", StringType(), True),
        StructField("flight_id", StringType(), True),
        StructField("counter", StringType(), True),
        StructField("gate", StringType(), True),
        StructField("departure_region_name", StringType(), True),
        StructField("arrival_region_name", StringType(), True),
        StructField("departure_airport", StringType(), True),
        StructField("arrival_airport", StringType(), True),
        StructField("airline", StringType(), True)
    ])
'''

class BookingSchema(BaseModel):
    user_name: str
    user_phone: str
    user_email: str
    date_book: str
    flight_id: str
    
class FlightReceipt(BaseModel):
    booking_info: BookingSchema
    flight_info: FlightSchema

class ConnectDB_Event(Event):
    payload: str 

class ConnectDBComplete_Event(Event):
    payload: str

class ParseInput_Event(Event):
    payload: str

class QueryGeneration_Event(Event):
    payload: FlightSchema | BookingSchema  

class PostProcessQuery_Event(Event):
    payload: str | dict

class QueryGenerationComplete_Event(Event):
    payload: str | dict
    
class RetrieveEvent(Event):
    payload: str

class CleanUp(Event):
    payload: str


class MongoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super().default(obj)
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from llama_index.embeddings.openai import OpenAIEmbedding

async def query_regulation(query: str) -> str:
        qdrant_client = QdrantClient(QDRANT_HOST)
        es = Elasticsearch(ELASTICSEARCH_HOST)
        embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL)
        
        query_embedding = embed_model.get_text_embedding(query)
        top_k = 3
        qdrant_results = qdrant_client.search(
            collection_name="regulations",
            query_vector=query_embedding,
            limit=top_k
        )

        es_results = es.search(
            index="regulations",
            body={
                "query": {
                    "match": {
                        "text": query
                    }
                },
                "size": top_k
            }
        )

        combined_results = []
        seen_texts = set()

        for hit in qdrant_results:
            text = hit.payload.get("text")
            if text not in seen_texts:
                combined_results.append({
                    "text": text,
                    "filename": hit.payload.get("filename"),
                    "score": hit.score,
                    "source": "qdrant"
                })
                seen_texts.add(text)

        for hit in es_results["hits"]["hits"]:
            text = hit["_source"]["text"]
            if text not in seen_texts:
                combined_results.append({
                    "text": text,
                    "filename": hit["_source"]["filename"],
                    "score": hit["_score"],
                    "source": "elasticsearch"
                })
                seen_texts.add(text)

        observation = '\n\n'.join([i['text'] for i in combined_results])
        return observation
        #print(f'RAG data: {observation}')

class MongoDBflow(Workflow):
    @step 
    async def start(self, ctx: Context, ev: StartEvent) -> ConnectDB_Event | QueryGeneration_Event :
        print('Retrieve flow')
        MONGO_DB = "flight_db"  
        COLLECTION_NAME = "flight_info"  
        CONNECTION_STRING = "mongodb://localhost:27017" 

        LLM = OpenAI(
            model='gpt-3.5-turbo',
            logprobs=None,  
            default_headers={},  
        )
    
        await ctx.set('MONGO_DB', MONGO_DB)
        await ctx.set('COLLECTION_NAME', COLLECTION_NAME)
        await ctx.set('CONNECTION_STRING', CONNECTION_STRING)
        await ctx.set('LLM', LLM)

        try:
            query_str = ev.query
            if isinstance(query_str, str):
                query_str = query_str.replace("'", '"')
            query_str = json.dumps(query_str)
            db_query = json.loads(query_str)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid query format: {query_str}") from e

        #print(f'Initial query: {db_query}')
        await ctx.set('mongoDB_query', db_query)

        ctx.send_event(ConnectDB_Event(payload=''))
        ctx.send_event(QueryGenerationComplete_Event(payload=''))

    @step
    async def checking_query(self, ctx: Context, ev: QueryGeneration_Event) -> QueryGenerationComplete_Event:
        # PERFORM CHECKING VALID QUERY
        # ensure all field are filled
        if not isinstance(ev.payload, dict):
            raise ValueError("Input is not valid query object")
        
        #ctx.send_event(QueryGenerationComplete_Event(payload='Query generated success'))
        return QueryGenerationComplete_Event(payload='Query generated success')
    
    @step
    async def connect_mongoClient(self, ctx: Context, ev: ConnectDB_Event) -> ConnectDBComplete_Event:
        try:
            connection_string = await ctx.get('CONNECTION_STRING')
            mongo_db = await ctx.get('MONGO_DB')
            collection_name = await ctx.get('COLLECTION_NAME')

            client = MongoClient(connection_string)
            await ctx.set('mongo_client' , client)

            db = client.get_database(name=mongo_db)
            collection = db[collection_name]

            await ctx.set('collection', collection)
            return ConnectDBComplete_Event(payload='Connect to database success')
        except Exception as e:
            raise ValueError(f"Failed to connect to MongoDB: {str(e)}")

    @step
    async def retrieve_mongoClient(self, ctx: Context, ev: ConnectDBComplete_Event | QueryGenerationComplete_Event ) -> CleanUp:
        
        #print("Received event: ", ev.payload)
        if ( ctx.collect_events( ev, [ConnectDBComplete_Event, QueryGenerationComplete_Event]) is None):
            return None
        try: 
            collection = await ctx.get('collection')
            mongoDB_query = await ctx.get('mongoDB_query')
            final_query = {}
            ### Apply retrieving logic with schemas
            for key, value in mongoDB_query.items():
                if key == 'departure_time' or key == 'arrival_time':
                    final_query[key] = { "$regex": f"^{value}"}
                else:
                    final_query[key] = value
            
            #print(f'Final Query: {final_query}')
            retrieved_data = collection.find(final_query).to_list(length=None)

            await ctx.set('retrieved_data', retrieved_data)

            return CleanUp(payload='Retrieving data success')
        except Exception as e:
            raise ValueError(f"Failed to retrieve data: {str(e)}")

    @step
    async def clean_up(self, ctx: Context, ev: CleanUp) -> StopEvent:
        client = await ctx.get('mongo_client', None)
        retrieved_data = await ctx.get('retrieved_data', None)

        print(f'Retrieved flow : {retrieved_data}')

        if client:
            client.close()
            print("MongoDB client closed from Retrieve flow.")
        
        if not retrieved_data: # empty list
            formatted_string = 'No data retrieved, tell user there is no flights available.'
        else: 
            formatted_string = "\n".join(json.dumps(item, indent=4, cls=MongoEncoder) for item in retrieved_data)
        
        #print(formatted_string)
        observation = f"Retrieved data: \n{formatted_string}"
        return StopEvent(result=observation)

#draw_all_possible_flows(MongoDBflow)
# - `route` (string): The flight's route, in the format **"from <destination1> to <destination2>"**. Both `destination1` and `destination2` must be provided. For example, "from NYC to LAX".
# "route": "from NYC to LAX",


class BookFlow(Workflow):
    @step 
    async def start(self, ctx: Context, ev: StartEvent) -> ConnectDB_Event | QueryGenerationComplete_Event :
        MONGO_DB = "flight_db"  
        COLLECTION_NAME = "booking_info"  
        CONNECTION_STRING = "mongodb://localhost:27017" 

        LLM = OpenAI(
            model='gpt-3.5-turbo',
            logprobs=None,  
            default_headers={},  
        )
    
        await ctx.set('MONGO_DB', MONGO_DB)
        await ctx.set('COLLECTION_NAME', COLLECTION_NAME)
        await ctx.set('CONNECTION_STRING', CONNECTION_STRING)
        await ctx.set('LLM', LLM)

        try:
            print('Book flow')
            query_str = ev.query
            #print(f'query_str: {query_str}, type: {type(query_str)}') 

            query_str = query_str.replace("'", '"')    
            db_query = json.loads(query_str)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid query format: {query_str}") from e

        #print(f'Initial query: {db_query}, type: {type(db_query)}')
        await ctx.set('mongoDB_query', db_query)

        ctx.send_event(ConnectDB_Event(payload=''))
        ctx.send_event(QueryGenerationComplete_Event(payload=''))

    @step
    async def connect_mongoClient(self, ctx: Context, ev: ConnectDB_Event) -> ConnectDBComplete_Event:
        try:
            connection_string = await ctx.get('CONNECTION_STRING')
            mongo_db = await ctx.get('MONGO_DB')
            collection_name = await ctx.get('COLLECTION_NAME')

            client = MongoClient(connection_string)
            await ctx.set('mongo_client' , client)

            db = client.get_database(name=mongo_db)
            collection = db[collection_name]

            await ctx.set('collection', collection)
            return ConnectDBComplete_Event(payload='Connect to database success')
        except Exception as e:
            raise ValueError(f"Failed to connect to MongoDB: {str(e)}")


    @step
    async def retrieve_mongoClient(self, ctx: Context, ev: ConnectDBComplete_Event | QueryGenerationComplete_Event) -> CleanUp:
        if ctx.collect_events(ev, [ConnectDBComplete_Event, QueryGenerationComplete_Event]) is None:
            return None
        try: 
            collection = await ctx.get('collection', None)
            mongoDB_query = await ctx.get('mongoDB_query', None)
            
            if collection is None or mongoDB_query is None :
                return CleanUp(payload='Submit booking data failed: Missing required context data')
            
            flight_id_val = mongoDB_query['flight_id']
            w = MongoDBflow(timeout=30, verbose=False)
            flight_info = await w.run(query= {'flight_id': flight_id_val})
            await ctx.set('flight_info', flight_info)

            #print(f'Flight Info: {flight_info}')
            #print(f'Final Query: {mongoDB_query}')
            
            collection.insert_one(mongoDB_query)
            return CleanUp(payload='Submit booking data success')
        except Exception as e:
            print(f"Error during booking submission: {str(e)}")
            return CleanUp(payload='Submit booking data failed')

    @step
    async def clean_up(self, ctx: Context, ev: CleanUp) -> StopEvent:
        client = await ctx.get('mongo_client', None)
        submit_form = await ctx.get('mongoDB_query', None)
        flight_info = await ctx.get('flight_info', None)

        if 'no flights available' in flight_info.lower():
            flight_info = 'USER CANNOT BOOK FLIGHT BECAUSE FLIGHT_ID DOES NOT EXIST!'

        print(f'Booking flow : {submit_form}')

        if client:
            client.close()
            print("MongoDB client closed from Booking flow.")
        
        if ev.payload == 'Submit booking data failed':  # empty list
            result = 'Booking submission has failed, please try again'
        else:
            if not submit_form:
                receipt_form = "No booking information available."
            else: 
                receipt_form = f"""
            User information: {submit_form},\nBooked flight information: {flight_info}
""".strip()
            result = f'Booking submission has succeeded. Booking Receipt: \n{receipt_form}\n'
        
        observation = f"\n{result}"
        return StopEvent(result=observation)

class IntentType(str, Enum):
    QUERY = "query"
    BOOKING = "booking"
    REGULATION = "regulation"
    UNKNOWN = "unknown"

class IntentClassification(BaseModel):
    intent: IntentType
    confidence: float = 1.0

intent_classification_prompt = PromptTemplate(prompts.PARSE_PROMPTS_INTENTION)

class Booking_Event(Event):
    payload: str

class GatherInformation(Workflow):
    @step
    async def start(self, ctx: Context, ev: StartEvent) -> ParseInput_Event | StopEvent | Booking_Event:
        LLM = OpenAI(
            model='gpt-3.5-turbo',
            logprobs=None,
            default_headers={},
        )
        await ctx.set('LLM', LLM)
        #print(f'User input: {ev.query}')
        
        intent_classification = LLM.structured_predict(
            IntentClassification,
            intent_classification_prompt,
            text=ev.query
        )
        
        await ctx.set('Intent', intent_classification.intent)

        print(f"Detected intent: {intent_classification.intent} with confidence: {intent_classification.confidence}")
        
        if intent_classification.intent == IntentType.QUERY or intent_classification.intent == IntentType.BOOKING:
            return ParseInput_Event(payload=ev.query) #  RETRIEVE TASK
        
        elif intent_classification.intent == IntentType.BOOKING:
            return Booking_Event(payload=ev.query) # BOOKING TASK
        
        elif intent_classification.intent == IntentType.REGULATION:
            regulation_prompt = PromptTemplate(prompts.PARSE_PROMPTS_REGULATION)
            regulation_response = LLM.complete(regulation_prompt, text=ev.query)
            return StopEvent(result=str(regulation_response)) # REGULATION TASK
        else:
            return StopEvent(result="I'm not sure what you're asking for. Could you please rephrase your question? You can ask me about flight schedules, airline regulations, or general flight information.")

    @step
    async def booking_action(self, ctx: Context, ev: Booking_Event) -> QueryGenerationComplete_Event:
        llm = await ctx.get('LLM')
        user_query = ev.payload

        simplified_prompt = PromptTemplate(prompts.PARSE_PROMPTS_BOOKING)

        extracted_input = llm.structured_predict( BookingSchema, simplified_prompt, text= user_query)
        #print(f'Book extracted: {extracted_input}')

        if not isinstance(extracted_input, BaseModel):
            raise ValueError("Parsed input is not a valid Pydantic object.")
        
        return StopEvent(result='Please provide more information')

    @step
    async def parse_userQuery(self, ctx: Context, ev: ParseInput_Event) -> QueryGeneration_Event:
        
        intent = await ctx.get('Intent')
        if intent == IntentType.QUERY:
            simplified_prompt = PromptTemplate(prompts.PARSE_PROMPTS_RETRIVE)
            target_schema = FlightSchema
        if intent == IntentType.BOOKING:
            simplified_prompt = PromptTemplate(prompts.PARSE_PROMPTS_BOOKING)
            target_schema = BookingSchema

        try:
            llm = await ctx.get('LLM')
            extracted_input = llm.structured_predict( target_schema, simplified_prompt, text=ev.payload)
            #print(f"Extracted input: {extracted_input}")
            return QueryGeneration_Event(payload=extracted_input)
        
        except Exception as e:
            raise ValueError(f"Failed to parse query: {str(e)}")
        
    @step
    async def generate_query(self, ctx: Context, ev: QueryGeneration_Event) -> PostProcessQuery_Event:
        
        #pydantic object
        parsed_input = ev.payload
        
        if not isinstance(parsed_input, BaseModel):
            raise ValueError("Parsed input is not a valid Pydantic object.")

        # Assuming 'model' is your Pydantic object
        query = parsed_input.model_dump(exclude_none=True)

        if not isinstance(query, dict):
            raise ValueError("Generated query is not a valid MongoDB query object.")
        
        #print(f"Initial Query: {query}")
        return PostProcessQuery_Event(payload=query)
        
    
    @step
    async def postprocess_query(self, ctx: Context, ev: PostProcessQuery_Event) -> QueryGenerationComplete_Event:
        
        def parseTime(time: str):
            parsed = dateparser.parse(time).strftime("%Y-%m-%d")
            '''
            if parsed.hour == 0 and parsed.minute==0:
                parsed = parsed.strftime("%Y-%m-%d")
            '''

            return parsed

        #print(ev.payload)
        
        tmp = ev.payload
        '''
        "departure_time": departure_time.strftime("%Y-%m-%d %H:%M"),
        "arrival_time": arrival_time.strftime("%Y-%m-%d %H:%M")
        '''
        
        departTime = tmp.get('departure_time', None)
        datebookTime = tmp.get('date_book', None)
        scheduleTime = tmp.get('scheduled_time', None)  # Default to None if the key doesn't exist
        updatedTime = tmp.get('updated_time', None)
        dateTime = tmp.get('date', None)

        if datebookTime is not None:
            parsedDatebook = parseTime(datebookTime)
            tmp['date_book'] = parsedDatebook

        if departTime is not None:
            parsedDepart = parseTime(departTime)
            tmp['departure_time'] = parsedDepart
        
        if dateTime is not None:
            parsedDateTime = parseTime(dateTime)
            tmp['date'] = parsedDateTime

        if scheduleTime is not None:
            parsedSchedule = parseTime(scheduleTime)
            tmp['scheduled_time'] = parsedSchedule
            
        if updatedTime is not None:
            parsedUpdated = parseTime(updatedTime)
            tmp['updated_time'] = parsedUpdated

        query = tmp
        
        #print(f"Final query: {query}")

        return QueryGenerationComplete_Event(payload=query)

    @step
    async def query_output(self, ctx: Context, ev: QueryGenerationComplete_Event) -> StopEvent:
        query = ev.payload
        intent = await ctx.get('Intent')
        observation = f'Query: {query}, Intent: {intent.value}.'
        return StopEvent(result=observation)


######################################################################################################
async def prepare_input(user_query: str) -> dict:
    w = GatherInformation(timeout=30, verbose=False)
    result = await w.run(query= user_query)
    return result

async def read_mongodb(mongodb_query: str) -> str: 
    w = MongoDBflow(timeout=30, verbose=False)
    result = await w.run(query= mongodb_query)
    return result

async def submit_booking(booking_query: str) -> str:
    w = BookFlow(timeout=30, verbose=False)
    result = await w.run(query=booking_query)
    return result

async def retrieve_regulation(user_query: str) -> str:
    RAGdata = await query_regulation(query=user_query)
    return RAGdata

from llama_index.agent.openai import OpenAIAgent
import nest_asyncio
nest_asyncio.apply()

async def main():
    # Define the MongoDB Retriever tool
    read_mongodb_tool = FunctionTool.from_defaults(
        async_fn=read_mongodb,
        name='MongoDB_Retriever',
        description=(
            'Tool to retrieve information from the MongoDB database. This tool is used after got a valid MongoDB query.'
        )
    )

    RAG_regulation_tool = FunctionTool.from_defaults(
        async_fn=retrieve_regulation,
        name='RegulationRAG_tool',
        description=(
            'Tool to retrive regulation information from database.'
        )
    )

    booking_submit_tool = FunctionTool.from_defaults(
        async_fn=submit_booking,
        name="SubmitBooking_Tool",
        description=(
            'Tool to submit booking information to MongoDB database. This tool is used to after got a valid MongoDB query.'
        )
    )

    # Define the Query Preparation tool
    prepare_input_tool = FunctionTool.from_defaults(
        async_fn=prepare_input,
        name='Query_Prep',
        description=(
            'Tool to generate a Query from user input. Ensure user input is complete and accurate before invoking this tool.'
        )
    )

    # Define the toolset (add more tools here if needed)
    tools = [prepare_input_tool, read_mongodb_tool, booking_submit_tool, RAG_regulation_tool]

    # Initialize the OpenAI language model
    llm = OpenAI(
        model='gpt-3.5-turbo',
        logprobs=None,  # Thêm tham số này
        default_headers={},  # Thêm tham số này
    )

    # Initialize the agent with tools and prompt
    agent = OpenAIAgent.from_tools(tools, llm=llm, verbose=True, system_prompt=prompts.SYSTEM_PROMPT)

    def cleanup():
        pass

    try:
        print("Flight assistant is ready to assist. Type your query below:")
        while True:
            user_input = input("USER: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye! Have a great day!")
                cleanup()
                break

            # Run the agent and handle user queries
            response = agent.chat(user_input)

            print(f"AGENT: {response}")
    except KeyboardInterrupt:
        print("\nExiting. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")
    
if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
