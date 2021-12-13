<template>
    <div class="page">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class='value' size="small" filterable v-model="params.model">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
            </div>
            <div class="item"><div class="title">target</div><el-input class='value' type='number' size="small" v-model="params.target" @change="update"/></div>
            <div class="item"><div class="title">epochs</div><el-input class='value' type='number' size="small" v-model="params.epochs" /></div>
            <div class="item"><div class="title">lr</div><el-input class='value' type='number' size="small" v-model="params.lr"  /></div>
            <div class="item"><el-checkbox class='button' v-model="params.blur">blur</el-checkbox></div>
            <div class="item"><div class="title">blur freq</div><el-input class='value' type='number' size="small" v-model="params.blur_freq" /></div>
            <div class="item"><div class="title">weight decay</div><el-input class='value' type='number' size="small" v-model="params.weight_decay" /></div>
            <div class="item"><el-checkbox class='button' v-model="params.clip_grad">clip grad</el-checkbox></div>
            <div class="item"><el-checkbox class='button' v-model="params.clamp">clamp</el-checkbox></div>
            <div class="item"><div class="title"></div><el-button icon='el-icon-refresh' type="primary" size="small" circle  @click="update"/></div>
        </div>
        <div class="network">
            <div class="iter">
                <img :src="res.output" width="256" height="256"/>
                <div>epoch = {{res.epoch}}, loss = {{res.loss}}</div>

                <div class="predictions">
                    <div class="item" v-for="cls in res.topk" :key="cls.index">
                        <div class="name">{{cls.index}}/{{cls.class}}</div><div class="value">{{cls.confidence}}</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    data() {
        return {
            models: [],
            res: {},
            params: {
                model: 'vgg19',
                target: 130,
                epochs: 300,
                lr: 3,
                clamp: false,
                blur: true,
                blur_freq: 2,
                weight_decay: 0,
                clip_grad: false,
            }
        };
    },
    created() {
        this.config()
    },
    sockets: {
        models(data) {
            this.models = data
        },
        response_class_max(data) {
            this.res = data
        }
    },
    methods: {
        config() {
            this.$socket.emit('get_models')
        },
        update() {
            this.$socket.emit("class_max", this.params);
        },
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network {
        .iter {
            display: flex;
            flex-flow: column;
            align-items: center;
        }

        .predictions {
            .item {
                display: flex;
                flex-flow: column;
                margin: 10px 0;
                align-items: center;
                justify-content: center;

                .name {
                    text-align: center;
                    color: #666;
                }

                .value {
                    color: #333;
                    font-weight: 600;
                }
            }
        }
    }
}

</style>
