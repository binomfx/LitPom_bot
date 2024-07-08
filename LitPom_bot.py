####################################################################################################
####################################################################################################
#######################################  LitPom_bot  ###############################################
####################################################################################################
####################################################################################################
from langchain.chat_models.gigachat import GigaChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

import os
import os.path
from os import path
from google.colab import userdata

amvera_var = 0
#amvera_var = os.environ["MY_VAR"]
if amvera_var == 1:
  # Импортировать ключ для авторизации в GigaChat и токен от Telegram бота
  sber = os.environ('SBER_AUTH')
  bot_token = os.environ('BOT_TOKEN')
  # Импортировать промпты
  r_prompt = os.environ('RAG_PROMPT')
  c_prompt = os.environ('CONV_PROMPT')
else:
  # Импортировать ключ для авторизации в GigaChat и токен от Telegram бота
  sber = userdata.get('SBER_AUTH')
  bot_token = userdata.get('BOT_TOKEN')
  # Импортировать промпты
  r_prompt = userdata.get('RAG_PROMPT')
  c_prompt = userdata.get('CONV_PROMPT')

####################################################################################################
#                                            Инициализация                                         #
####################################################################################################

user_conversations = {} # Словарь для хранения ConversationBufferMemory каждого пользователя
user_llm_rag = {} # Словарь для хранения модели и rag каждого пользователя

doc_store = 'data' # Путь для сохранения векторных хранилищ

# Создать промпты
rag_prompt = ChatPromptTemplate.from_template('''Ответь на вопрос пользователя. \
Используй при этом только информацию из контекста. Это Важно! Если в контексте нет \
информации для ответа, сообщи об этом пользователю фразой - в контексте нет необходимой информации.
Контекст: {context}
Вопрос: {input}
Ответ:'''
)
conv_prompt = c_prompt

# Создать объект бота
import telebot
from time import sleep
from telebot import types

bot = telebot.TeleBot(bot_token)


####################################################################################################
#                                       Генеративная часть                                         #
####################################################################################################

def create_llm_rag(user_id):
    # Создать объект LLM GigaChat
    llm = GigaChat(credentials=sber,
              model='GigaChat:latest',
               verify_ssl_certs=False,
               profanity_check=False)
    # Создать эмбеддер
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(model_name=model_name,
                                  model_kwargs=model_kwargs,
                                  encode_kwargs=encode_kwargs)
    # Создать векторное хранилище
    if path.exists(doc_store + '/' + str(user_id) + ".faiss"):
      # Загрузить существующее векторное хранилище пользователя
      bot.send_message(user_id, 'Загрузка векторного хранилища')
      vector_store = FAISS.load_local(folder_path=doc_store, embeddings=embedding, index_name=str(user_id),
                        allow_dangerous_deserialization=True )
    else:
      # Создать пустое векторное хранилище
      bot.send_message(user_id, 'Создание векторного хранилища')
      texts = ["FAISS is an important library", "LangChain supports FAISS"]
      vector_store = FAISS.from_texts(texts, embedding)
    # Создать ретривер
    embedding_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    # Создать вопросно-ответную цепочку RAG
    rag_chain = create_rag_chain(user_id, llm, embedding_retriever)
    # Создать цепочку диалога
    conversation_chain = create_conversation_chain(user_id, llm)
    return (vector_store, embedding_retriever, llm, rag_chain, conversation_chain)

# Создать вопросно-ответную цепочку RAG
####################################################
def create_rag_chain(user_id, llm, embedding_retriever):
    # (1)
    # Создадим цепочку create_stuff_documents_chain, которая является частью
    # вопросно-ответной системы, ответственной за вставку фрагментов текстов
    # из векторной БД в промпт языковой модели
    document_chain = create_stuff_documents_chain(
      llm=llm,
      prompt=rag_prompt
      )
    # Создадим вопросно-ответную цепочку с помощью функции create_retrieval_chain().
    # используем ретривет для векторной базы с книгой
    retrieval_chain = create_retrieval_chain(embedding_retriever, document_chain)
    # Обращение к системе
    #resp1 = retrieval_chain.invoke(
    #    {'input': q1}
    #)
    return (retrieval_chain)

# Создать цепочку диалога
####################################################
def create_conversation_chain(user_id, llm):
    # (2)
    # Создать объект цепочки диалога - инициализация ConversationChain
    conversation = ConversationChain(llm=llm,
                                 verbose=True,
                                 memory=ConversationBufferMemory())
    conversation.prompt.template = conv_prompt # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< Промпт диалога
    # Обращение к системе
    # conversation.predict(input='Как меня зовут и чем я занимаюсь?')
    return(conversation)

####################################################
# Загрузка и векторизация текста
def learn_document(doc_file,vector_store):
    # ЗАГРУЗКА И НАРЕЗКА ТЕКСТА DOCX
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024,
                                              chunk_overlap=128)
    # Определить расшиение файла - docx, pdf или fb2
    file_ext = doc_file[doc_file.rfind(".") + 1:]
    # Использовать соответствующий загрузчик
    if file_ext == 'docx':
      loader = UnstructuredWordDocumentLoader(doc_file)
    elif file_ext == 'pdf':
      loader = UnstructuredPDFLoader(doc_file)
    elif file_ext == 'fb2':
      loader = UnstructuredXMLLoader(doc_file)
    else:
      return (0)

    splitted_data = loader.load_and_split(text_splitter)
    # ВЕКТОРИЗАЦИЯ
    # Добавить документ в векторное хранилище
    ids_list = vector_store.add_documents(splitted_data)

    return(len(ids_list))

####################################################################################################
#                                    Функции обработки команд                                      #
####################################################################################################

# `/start` - функция, обрабатывающая команду
#############################################
@bot.message_handler(commands=['start'])
def start(message: types.Message):
    user_id = message.chat.id

    # Проверка словарей для данного пользователя
    if user_id not in user_conversations:
        user_conversations[user_id] = ConversationBufferMemory()

    if user_id not in user_llm_rag:
        user_llm_rag[user_id] = create_llm_rag(user_id)

    vdb, embedding_retriever, llm, rag_chain, conversation = user_llm_rag[user_id]
    conversation.memory = user_conversations[user_id]

    bot.send_message(message.chat.id, 'Готов к работе')

# `/help` - функция, обрабатывающая команду
#############################################
@bot.message_handler(commands=['help'])
def help(message: types.Message):
    user_id = message.chat.id

    bot.send_message(message.chat.id, 'Я - бот-помощник в работе с текстами. Вы можете отправить мне файлы в форматах .doc .pdf .fb2, я их обработаю, загружу в векторную базу данных и Вы сможете задавать мне вопросы по этим текставм.')

####################################################################################################
#                                  Функции обработки сообщений                                     #
####################################################################################################

# Функция, обрабатывающая неправильные форматы ввода
####################################################
@bot.message_handler(content_types=['audio',
                                    'video',
                                    'photo',
                                    'sticker',
                                    'voice',
                                    'location',
                                    'contact'])
def not_text(message):
  user_id = message.chat.id
  bot.send_message(user_id, 'Я работаю только с текстовыми сообщениями и документами!')

# Функция, обрабатывающая файлы документов
##########################################
@bot.message_handler(content_types=['document'])
def handle_doc_message(message):
    user_id = message.chat.id

    # Проверка словарей для данного пользователя
    if user_id not in user_conversations:
        user_conversations[user_id] = ConversationBufferMemory()

    if user_id not in user_llm_rag:
        user_llm_rag[user_id] = create_llm_rag(user_id)

    vdb, embedding_retriever, llm, rag_chain, conversation = user_llm_rag[user_id]
    conversation.memory = user_conversations[user_id]

    # Загрузка файла
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    file_name = message.document.file_name
    bot.reply_to(message, "Читаю документ: " + file_name)
    # src = 'C:/Python/Project/tg_bot/files/received/' + file_name;
    with open(file_name, 'wb') as new_file:
        new_file.write(downloaded_file)

    # Векторизация документа и добавление в векторное хранилище пользователя
    r = learn_document(file_name, vdb)
    # Сообщение пользователю о результате операции
    if r > 0:
      bot.send_message(user_id, "Документ прочитан ("+ str(r) +')')
      # Сохранить дополненное векторное хранилище
      vdb.save_local(doc_store, str(user_id))
    else:
      bot.send_message(user_id, "Не могу прочитать документ, вероятно не понятный формат.")

    sleep(2)


# Функция, обрабатывающая текстовые сообщения
#############################################
@bot.message_handler(content_types=['text'])
def handle_text_message(message):
    user_id = message.chat.id

    # Проверка словарей для данного пользователя
    if user_id not in user_conversations:
        user_conversations[user_id] = ConversationBufferMemory()

    if user_id not in user_llm_rag:
        user_llm_rag[user_id] = create_llm_rag(user_id)

    vdb, embedding_retriever, llm, rag_chain, conversation = user_llm_rag[user_id]
    conversation.memory = user_conversations[user_id]

    # Получение и отправка ответа через GigaChat
    q1 = message.text
    # (RAG)
    resp1 = rag_chain.invoke(
        {'input': q1}
    )
    bot.send_message(user_id, 'RAG ('+ q1 +'):')
    answer = resp1['answer']
    bot.send_message(user_id, answer)
    # (LLM)
    # q2 = answer
    # resp2 = conversation.predict(input=q2)
    # bot.send_message(user_id, 'LLM:')
    # bot.send_message(user_id, conversation.memory.chat_memory.messages[-1].content)

    # ........

    sleep(2)

####################################################################################################
#                                      Запуск бота                                                 #
####################################################################################################
bot.polling(none_stop=True)
