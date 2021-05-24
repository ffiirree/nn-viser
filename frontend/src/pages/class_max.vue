<template>
    <div class="page">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class='value' size="small" v-model="params.model" @change="update">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
            </div>
            <div class="item"><div class="title">target</div><el-input class='value' type='number' size="small" v-model="params.target"  @change="update"/></div>
            <div class="item"><div class="title">epochs</div><el-input class='value' type='number' size="small" v-model="params.epochs"  @change="update"/></div>
            <div class="item"><div class="title">lr</div><el-input class='value' type='number' size="small" v-model="params.lr"  @change="update"/></div>
            <div class="item"><el-checkbox class='button' v-model="params.blur" @change="update">blur</el-checkbox></div>
            <div class="item"><div class="title">blur freq</div><el-input class='value' type='number' size="small" v-model="params.blur_freq"  @change="update"/></div>
            <div class="item"><div class="title">weight decay</div><el-input class='value' type='number' size="small" v-model="params.weight_decay"  @change="update"/></div>
            <div class="item"><el-checkbox class='button' v-model="params.clip_grad" @change="update">clip grad</el-checkbox></div>
            <div class="item"><el-checkbox class='button' v-model="params.clamp" @change="update">clamp</el-checkbox></div>
        </div>
        <div class="network">
            <div class="iter">
                <img :src="res.output" width="256" height="256"/>
                <div>epoch = {{res.epoch}}, loss = {{res.loss}}</div>
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
                model: 'alexnet',
                target: 130,
                epochs: 225,
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
        this.update()
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
    }
}

</style>
