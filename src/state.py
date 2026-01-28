from typing import Annotated,List
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from enum import Enum

# 常量定义
Q_ROUNDS = 3
E_ROUNDS = 3
F_ROUNDS = 2

# 先定义若干枚举类
class ProviderEnum(str, Enum):
    PROSECUTOR = "prosecutor"
    DEFENDANT = "defendant"

class PhaseEnum(str, Enum):
    OPENING = "opening"                 # 开庭阶段
    INVESTIGATION = "investigation"     # 法庭调查
    DEBATE = "debate"                   # 法庭辩论
    VERDICT = "verdict"                 # 宣布判决

class Evidence_Show_Enum(str,Enum):
    SINGLE = "single"                       # 单一证据举证
    UNION = "union"                     # 联合证据举证
    QUIT = "quit"                       # 放弃举证

# 1. 定义证据结构
class Evidence(BaseModel):
    id: str                                 # 证据编号
    name: str                               # 证据名称
    content: str                            # 证据内容,为了简化，暂时先指定content为字符串类型，未来如果要处理多模态的数据，再进一步扩展
    provider: ProviderEnum                  # 证据提供者



# 2. 定义案件相关元数据信息
class case_info(BaseModel):
    # -- 案件本身信息 --
    abstract : str                                      # 案件摘要

    # -- 控方信息 --
    '''
    同样是考虑简化，起诉书先处理为字符串的形式
    '''
    prosecutor_title : str      # 公诉机关名
    prosecutor_name : str       # 代理检察员
    statement_charge : str      # 控方提交的起诉书，需与被告核对
    crime : str                 # 罪名

    # -- 辩方信息 --
    defendant_name : str                                # 被告人姓名
    defendant_former_name : str | None = None           # 被告人曾用名
    defendant_birthdate : str                           # 被告人出生日期（年月日）
    defendant_birthplace : str                          # 被告人出生地
    defendant_ethnicity : str                           # 被告人民族
    defendant_education : str                           # 被告人文化程度
    defendant_occupation : str                          # 被告人职业
    defendant_employer : str                            # 被告人工作单位
    defendant_residence : str                           # 被告人家庭住址
    defendant_ID_number : str                           # 被告人身份证号
    defendant_legal_record : str                        # 是否受过法律处分？
    detention_date : str                                # 因本案何时被羁押？
    indictment_date : str                               # 何时收到起诉书副本

    attorney_name : str                                 # 辩方律师姓名

    # -- 审理法院信息 --
    court_name : str        # 审理法院名
    judge_name : str        # 审判长姓名
    judge_name_2 : str      # 人民陪审员姓名
    clerk_name : str        # 书记员姓名
    case_id : str           # 卷宗代号

# 3. 定义整个庭审的状态
class CourtState(BaseModel):
    # --- 基础记忆 ---
    messages: Annotated[List[HumanMessage | AIMessage],add_messages]   # 对话历史 (HumanMessage, AIMessage)
    focus: list[str]                        # 审判长总结提出的争议焦点
    # --- 流程控制 ---
    phase: PhaseEnum                        # 当前法庭阶段
    
    # --- 案卷数据 ---
    evidence_list: List[Evidence]                           # 所有的证据
    current_evidence : List[Evidence] | None                # 当前公诉人拟举证的证据，可以为None
    evidence_show_type : Evidence_Show_Enum                 # 当前公诉人拟进行举证的模式：单一证据、联合证据、放弃举证
    meta : case_info                                        # 案件相关的元数据信息

    # --- 缓存中间消息 ---
    pros_question_rounds: int = Q_ROUNDS           # 用来缓存公诉人对被告人的提问剩余次数，默认提问三次
    pros_evidence_rounds: int = E_ROUNDS           # 用来缓存公诉人举证的最大次数，默认举证三次
    pros_focus_rounds: int = F_ROUNDS              # 用来缓存法庭辩论时争议焦点解决的最大轮数，默认回应两轮
    focus_index: int = 0                           # 用来缓存当前争议焦点的索引
